"""
Reproduce gradient scale differences when computing gradients separately vs together.

Run with: python examples/preconditioner_instability.py
"""

import subprocess
from pathlib import Path

import torch
from datasets import Dataset

from bergson.data import load_gradients


def run_bergson_build(index_path: str, dataset_path: str):
    cmd = [
        "bergson", "build", index_path,
        "--model", "Qwen/Qwen3-4B",
        "--dataset", dataset_path,
        "--prompt_column", "text",
        "--fsdp",
        "--projection_dim", "16",
        "--token_batch_size", "4000",
        "--overwrite",
    ]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main():
    # Create a simple dataset - just split some text in two
    # Need enough samples for 8 workers to each get at least one batch
    texts_a = [f"The quick brown fox jumps over the lazy dog {i}" for i in range(2000)]
    texts_b = [f"A journey of a thousand miles begins with a single step {i}" for i in range(2000)]

    ds_a = Dataset.from_dict({"text": texts_a})
    ds_b = Dataset.from_dict({"text": texts_b})
    ds_combined = Dataset.from_dict({"text": texts_a + texts_b})

    # Save datasets
    ds_a.save_to_disk("runs/instability_test/data_a")
    ds_b.save_to_disk("runs/instability_test/data_b")
    ds_combined.save_to_disk("runs/instability_test/data_combined")

    # Build three indices
    print("=" * 70)
    print("Building index A (first half)...")
    run_bergson_build("runs/instability_test/a", "runs/instability_test/data_a")

    print("=" * 70)
    print("Building index B (second half)...")
    run_bergson_build("runs/instability_test/b", "runs/instability_test/data_b")

    print("=" * 70)
    print("Building combined index...")
    run_bergson_build("runs/instability_test/combined", "runs/instability_test/data_combined")

    # Compare gradient scales
    grads_a = torch.from_numpy(
        load_gradients(Path("runs/instability_test/a"), structured=False).copy()
    ).float()
    grads_b = torch.from_numpy(
        load_gradients(Path("runs/instability_test/b"), structured=False).copy()
    ).float()
    grads_combined = torch.from_numpy(
        load_gradients(Path("runs/instability_test/combined"), structured=False).copy()
    ).float()

    # Split combined to match a and b
    grads_a_in_combined = grads_combined[:2000]
    grads_b_in_combined = grads_combined[2000:]

    print()
    print("=" * 70)
    print("GRADIENT SCALE COMPARISON")
    print("=" * 70)
    print(f"A (separate):      std = {grads_a.std():.4e}")
    print(f"A (from combined): std = {grads_a_in_combined.std():.4e}")
    print(f"Ratio: {grads_a.std() / grads_a_in_combined.std():.2f}x")
    print()
    print(f"B (separate):      std = {grads_b.std():.4e}")
    print(f"B (from combined): std = {grads_b_in_combined.std():.4e}")
    print(f"Ratio: {grads_b.std() / grads_b_in_combined.std():.2f}x")

    # Cosine similarity - are they the same direction?
    a_norm = grads_a / grads_a.norm(dim=1, keepdim=True)
    a_comb_norm = grads_a_in_combined / grads_a_in_combined.norm(dim=1, keepdim=True)
    cosines = (a_norm * a_comb_norm).sum(dim=1)
    print()
    print(f"Cosine similarity (A separate vs A in combined): mean={cosines.mean():.4f}")


if __name__ == "__main__":
    main()
