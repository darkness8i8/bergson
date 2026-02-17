"""Tests for advantage estimation in the data pipeline."""

import json

import pytest
from datasets import Dataset

from bergson.config import DataConfig, IndexConfig
from bergson.utils.worker_utils import estimate_advantage, setup_data_pipeline

PYTHIA = "EleutherAI/pythia-14m"


def create_rewards_file(tmp_path, prompts_and_rewards):
    """Create a JSON file with text and reward columns."""
    data = [
        {"text": text, "reward": reward}
        for text, reward in prompts_and_rewards
    ]
    path = tmp_path / "rewards.json"
    path.write_text("\n".join(json.dumps(d) for d in data))
    return str(path)


def test_estimate_advantage_single_group():
    """Advantages within a single prompt group should sum to zero."""
    ds = Dataset.from_dict(
        {
            "prompt": ["hello", "hello", "hello"],
            "reward": [1.0, 2.0, 3.0],
        }
    )
    cfg = DataConfig(prompt_column="prompt", reward_column="reward")
    advantages = estimate_advantage(ds, cfg)

    assert len(advantages) == 3
    assert abs(sum(advantages)) < 1e-9
    # Mean reward = 2.0; expected advantages = [-1.0, 0.0, 1.0]
    assert abs(advantages[0] - (-1.0)) < 1e-9
    assert abs(advantages[1] - 0.0) < 1e-9
    assert abs(advantages[2] - 1.0) < 1e-9


def test_estimate_advantage_multiple_groups():
    """Advantages are computed per-group, not globally."""
    ds = Dataset.from_dict(
        {
            "prompt": ["hello", "hello", "world", "world"],
            "reward": [1.0, 3.0, 10.0, 20.0],
        }
    )
    cfg = DataConfig(prompt_column="prompt", reward_column="reward")
    advantages = estimate_advantage(ds, cfg)

    assert len(advantages) == 4
    # "hello" group mean = 2.0: advantages = [-1.0, 1.0]
    assert abs(advantages[0] - (-1.0)) < 1e-9
    assert abs(advantages[1] - 1.0) < 1e-9
    # "world" group mean = 15.0: advantages = [-5.0, 5.0]
    assert abs(advantages[2] - (-5.0)) < 1e-9
    assert abs(advantages[3] - 5.0) < 1e-9


def test_advantages_computed_with_drop_columns(tmp_path):
    """Regression test for issue #96: advantage column is present in the output
    even when drop_columns=True removes the original reward column."""
    # Two prompts, each with two completions and different rewards.
    prompts_and_rewards = [
        ("hello world", 1.0),
        ("hello world", 3.0),
        ("foo bar", 10.0),
        ("foo bar", 20.0),
    ]
    dataset_path = create_rewards_file(tmp_path, prompts_and_rewards)

    cfg = IndexConfig(
        run_path=str(tmp_path / "run"),
        model=PYTHIA,
        token_batch_size=2048,
        drop_columns=True,
        data=DataConfig(
            dataset=dataset_path,
            reward_column="reward",
        ),
    )

    ds = setup_data_pipeline(cfg)

    # The reward column should have been dropped.
    assert "reward" not in ds.column_names

    # The advantage column must be present and have correct values.
    assert "advantage" in ds.column_names
    advantages = ds["advantage"]
    assert len(advantages) == 4
    # "hello world" group mean = 2.0: advantages = [-1.0, 1.0]
    assert abs(advantages[0] - (-1.0)) < 1e-9
    assert abs(advantages[1] - 1.0) < 1e-9
    # "foo bar" group mean = 15.0: advantages = [-5.0, 5.0]
    assert abs(advantages[2] - (-5.0)) < 1e-9
    assert abs(advantages[3] - 5.0) < 1e-9
