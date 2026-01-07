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
    # Skip if index already exists
    if Path(index_path).exists() and (Path(index_path) / "info.json").exists():
        print(f"Index {index_path} already exists, skipping...")
        return

    cmd = [
        "bergson", "build", index_path,
        "--model", "Qwen/Qwen3-4B",
        "--dataset", dataset_path,
        "--prompt_column", "text",
        "--fsdp",
        "--projection_dim", "16",
        "--token_batch_size", "4000",
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

    # Check if driven by outliers
    print()
    print("=" * 70)
    print("ROOT CAUSE: OUTLIERS")
    print("=" * 70)

    norms_a = grads_a.norm(dim=1)
    norms_a_comb = grads_a_in_combined.norm(dim=1)
    norms_b = grads_b.norm(dim=1)
    norms_b_comb = grads_b_in_combined.norm(dim=1)

    # Find the worst outliers
    ratio_a = norms_a / norms_a_comb
    ratio_b = norms_b / norms_b_comb
    worst_a_idx = ratio_a.argmax().item()
    worst_b_idx = ratio_b.argmin().item()  # B outlier has ratio < 1

    print(f"A outlier: sample {worst_a_idx}")
    print(f"  separate norm: {norms_a[worst_a_idx]:.2e}")
    print(f"  combined norm: {norms_a_comb[worst_a_idx]:.2e}")
    print(f"  ratio: {ratio_a[worst_a_idx]:.0f}x")
    print()
    print(f"B outlier: sample {worst_b_idx}")
    print(f"  separate norm: {norms_b[worst_b_idx]:.2e}")
    print(f"  combined norm: {norms_b_comb[worst_b_idx]:.2e}")
    print(f"  ratio: {ratio_b[worst_b_idx]:.4f}x (combined {1/ratio_b[worst_b_idx]:.0f}x larger)")

    # Filter outliers and recompute
    print()
    print("=" * 70)
    print("AFTER REMOVING OUTLIERS (norm > 10x median in EITHER index)")
    print("=" * 70)
    mask_a = (norms_a < 10 * norms_a.median()) & (norms_a_comb < 10 * norms_a_comb.median())
    mask_b = (norms_b < 10 * norms_b.median()) & (norms_b_comb < 10 * norms_b_comb.median())

    print(f"A: removed {(~mask_a).sum().item()} outliers")
    print(f"  std ratio: {grads_a[mask_a].std() / grads_a_in_combined[mask_a].std():.2f}x")
    print()
    print(f"B: removed {(~mask_b).sum().item()} outliers")
    print(f"  std ratio: {grads_b[mask_b].std() / grads_b_in_combined[mask_b].std():.2f}x")


if __name__ == "__main__":
    main()
