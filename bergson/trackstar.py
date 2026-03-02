from copy import deepcopy

from .build import build
from .config import (
    IndexConfig,
    PreprocessConfig,
    ScoreConfig,
    TrackstarConfig,
)
from .process_grads import mix_preconditioners
from .score.score import score_dataset
from .utils.worker_utils import validate_run_path


def trackstar(
    index_cfg: IndexConfig,
    score_cfg: ScoreConfig,
    preprocess_cfg: PreprocessConfig,
    trackstar_cfg: TrackstarConfig,
):
    """Run the full trackstar pipeline: preconditioners -> mix -> build -> score."""
    run_path = index_cfg.run_path
    value_precond_path = f"{run_path}/value_preconditioner"
    query_precond_path = f"{run_path}/query_preconditioner"
    mixed_precond_path = f"{run_path}/mixed_preconditioner"
    query_path = f"{run_path}/query"
    scores_path = f"{run_path}/scores"

    # Step 1: Compute normalizers and preconditioners on value dataset
    print("Step 1/5: Computing normalizers and preconditioners on value dataset...")
    value_precond_cfg = deepcopy(index_cfg)
    value_precond_cfg.run_path = value_precond_path
    value_precond_cfg.skip_index = True
    value_precond_cfg.skip_preconditioners = False
    validate_run_path(value_precond_cfg)
    build(value_precond_cfg, preprocess_cfg)

    # Step 2: Compute normalizers and preconditioners on query dataset
    print("Step 2/5: Computing normalizers and preconditioners on query dataset...")
    query_precond_cfg = deepcopy(index_cfg)
    query_precond_cfg.run_path = query_precond_path
    query_precond_cfg.data = trackstar_cfg.query
    query_precond_cfg.skip_index = True
    query_precond_cfg.skip_preconditioners = False
    validate_run_path(query_precond_cfg)
    build(query_precond_cfg, preprocess_cfg)

    # Step 3: Mix query and value preconditioners
    print("Step 3/5: Mixing preconditioners...")
    mix_preconditioners(
        query_path=query_precond_path,
        index_path=value_precond_path,
        output_path=mixed_precond_path,
        mixing_coefficient=trackstar_cfg.mixing_coefficient,
    )

    # Step 4: Build per-item query gradient index
    print("Step 4/5: Building query gradient index...")
    query_cfg = deepcopy(index_cfg)
    query_cfg.run_path = query_path
    query_cfg.data = trackstar_cfg.query
    query_cfg.processor_path = query_precond_path
    query_cfg.skip_preconditioners = True
    validate_run_path(query_cfg)
    build(query_cfg, preprocess_cfg)

    # Step 5: Score value dataset against query using mixed preconditioner
    print("Step 5/5: Scoring value dataset...")
    score_index_cfg = deepcopy(index_cfg)
    score_index_cfg.run_path = scores_path
    score_index_cfg.processor_path = value_precond_path
    score_index_cfg.skip_preconditioners = True
    score_cfg.query_path = query_path
    preprocess_cfg.preconditioner_path = mixed_precond_path
    validate_run_path(score_index_cfg)
    score_dataset(score_index_cfg, score_cfg, preprocess_cfg)
