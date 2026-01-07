import json
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset, concatenate_datasets, load_from_disk
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from bergson import load_gradient_dataset
from bergson.gradients import GradientProcessor


def reword(dataset, model_name: str, prompt_template: str, batch_size: int = 8):
    device = "cuda:3"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # REQUIRED for batched generation with Llama/Qwen/Mistral
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    model.eval()

    new_facts = []
    new_reworded = []

    # Convert dataset to list for easy slicing
    # (Assuming the dataset is small enough to fit in RAM, which 1000 items is)
    data_list = list(dataset)

    print(f"Starting generation with batch size: {batch_size}...")

    for i in tqdm(range(0, len(data_list), batch_size)):
        # 1. Prepare the batch
        batch_items = data_list[i : i + batch_size]
        prompts = [prompt_template.format(fact=item["fact"]) for item in batch_items]

        # 2. Tokenize (Batch mode)
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
        input_len = inputs.input_ids.shape[1]

        # 3. Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True,
                temperature=0.7,
                top_p=0.8,
                min_p=0.0,
            )

        # 4. Slice output to remove prompt (all at once)
        # With left-padding, the prompt is always the first 'input_len' tokens
        generated_tokens = outputs[:, input_len:]

        # 5. Decode batch
        decoded_batch = tokenizer.batch_decode(
            generated_tokens, skip_special_tokens=True
        )

        # 6. Store results
        for item, output_text in zip(batch_items, decoded_batch):
            new_facts.append(item["fact"])
            new_reworded.append(output_text.strip())

    # Reconstruct dataset
    return Dataset.from_dict({"fact": new_facts, "reworded": new_reworded})


def create_data():
    dataset = load_from_disk("data/facts_dataset.hf")

    for model_name in ["Qwen/Qwen3-8B-Base", "Meta-Llama/Meta-Llama-3-8B"]:
        model_short = model_name.split("/")[-1]

        # 1. Shakespeare
        shake_path = f"data/facts_dataset_shakespeare-{model_short}.hf"
        if not Path(shake_path).exists():
            prompt_shake = (
                "Reword the following fact in a Shakespearean style, adding flair and "
                "poetry.\n"
                "Do not include other text in your response, just the contents of the "
                "reworded fact.\n"
                "Fact: {fact}\n"
                "Your rewrite:"
            )

            ds_shake = reword(dataset, model_name, prompt_shake, batch_size=8)
            ds_shake.save_to_disk(shake_path)
            print("Shakespearean processing done.")

        # 2. Pirate
        pirate_path = f"data/facts_dataset_pirate-{model_short}.hf"
        if not Path(pirate_path).exists():
            prompt_pirate = (
                "Reword the following fact like it's coming from a pirate. Be creative!\n"
                "Do not include any other text in your response, just the contents of the "
                "reworded fact.\n"
                "Fact: {fact}\n"
                "Your rewrite:"
            )

            ds_pirate = reword(dataset, model_name, prompt_pirate, batch_size=8)
            ds_pirate.save_to_disk(pirate_path)
            print("Pirate processing done.")


def load_scores_matrix(scores_path: Path) -> np.ndarray:
    """Load the scores matrix from bergson score output."""
    with open(scores_path / "info.json") as f:
        info = json.load(f)

    num_items = info["num_items"]
    num_scores = info["num_scores"]

    # Handle both tuple format (from bergson) and list format (from JSON serialization)
    dtype_spec = info["dtype"]
    if isinstance(dtype_spec, list) and len(dtype_spec) > 0 and isinstance(dtype_spec[0], list):
        # Convert list of lists back to list of tuples
        dtype_spec = [tuple(item) for item in dtype_spec]

    scores_mmap = np.memmap(
        scores_path / "scores.bin",
        dtype=np.dtype(dtype_spec),
        mode="r",
        shape=(num_items,),
    )

    # Extract score columns into a dense matrix
    scores = np.zeros((num_items, num_scores), dtype=np.float32)
    for i in range(num_scores):
        scores[:, i] = scores_mmap[f"score_{i}"]

    return scores


def build_style_lookup(include_llama: bool = False) -> dict[tuple[str, str], str]:
    """Build a lookup from (fact, reworded) -> style name."""
    style_lookup = {}
    style_datasets = [
        ("data/facts_dataset_shakespeare-Qwen3-8B-Base.hf", "shakespeare"),
        ("data/facts_dataset_pirate-Qwen3-8B-Base.hf", "pirate"),
    ]
    if include_llama:
        style_datasets.extend([
            ("data/facts_dataset_shakespeare-Meta-Llama-3-8B.hf", "shakespeare-llama"),
            ("data/facts_dataset_pirate-Meta-Llama-3-8B.hf", "pirate-llama"),
        ])
    for path, style_name in style_datasets:
        ds = load_from_disk(path)
        for row in ds:
            style_lookup[(row["fact"], row["reworded"])] = style_name
    return style_lookup


def compute_metrics_groupwise(
    index_path: str,
    group_by: str = "field",  # "field" or "style"
    unit_normalize: bool = True,
):
    """Compute intra/inter similarities using group-aggregated gradients.

    Groups by either field (birthdate, employer, etc.) or style (shakespeare, pirate).
    Only uses Qwen styles (excludes Llama).

    Args:
        index_path: Path to the gradient index
        group_by: "field" or "style" - what to group by
        unit_normalize: Whether to unit normalize gradients
    """
    index_path = Path(index_path)

    # Load gradient dataset with metadata
    print("Loading gradient dataset...")
    grad_ds = load_gradient_dataset(index_path, structured=True)
    print(f"  Loaded {len(grad_ds)} rows")

    # Get gradient column names
    with open(index_path / "info.json") as f:
        info = json.load(f)
    grad_columns = info["dtype"]["names"]
    print(f"  Gradient columns: {len(grad_columns)} modules")

    # Build style lookup (Qwen only, no Llama)
    print("Building style lookup (Qwen only)...")
    style_lookup = build_style_lookup(include_llama=False)

    # Use batch column access for speed
    facts = grad_ds["fact"]
    reworded = grad_ds["reworded"]
    fields = grad_ds["field"]

    print("Mapping styles...")
    styles = [style_lookup.get((f, r), None) for f, r in zip(facts, reworded)]

    # Filter to only Qwen styles (exclude Llama and unknown)
    print("Filtering to Qwen styles only...")
    keep_indices = [i for i, s in enumerate(styles) if s is not None]
    grad_ds = grad_ds.select(keep_indices)
    styles = [styles[i] for i in keep_indices]
    fields = [fields[i] for i in keep_indices]
    print(f"  Keeping {len(grad_ds)} rows")

    # Build group keys based on group_by parameter
    print(f"Building groups by {group_by}...")
    if group_by == "field":
        group_keys = fields
    elif group_by == "style":
        group_keys = styles
    else:
        raise ValueError(f"group_by must be 'field' or 'style', got {group_by}")

    unique_groups = sorted(set(group_keys))
    group_to_idx = {g: i for i, g in enumerate(unique_groups)}
    row_to_group = torch.tensor([group_to_idx[g] for g in group_keys])
    print(f"  Found {len(unique_groups)} unique groups: {unique_groups}")

    # Load gradients directly from memmap (much faster than HF dataset)
    print("Loading gradients from memmap...")
    from bergson.data import load_gradients
    grad_mmap = load_gradients(index_path, structured=False)
    # Select only the kept rows
    all_grads = torch.from_numpy(grad_mmap[keep_indices].copy()).float()
    print(f"  Gradient tensor shape: {all_grads.shape}")

    # Compute mean gradient per group
    print("Computing mean gradients per group...")
    num_groups = len(unique_groups)
    group_grads = torch.zeros(num_groups, all_grads.shape[1], dtype=torch.float32)
    group_counts = torch.zeros(num_groups, dtype=torch.float32)

    for g_idx in range(num_groups):
        mask = row_to_group == g_idx
        group_grads[g_idx] = all_grads[mask].sum(dim=0)
        group_counts[g_idx] = mask.sum().float()

    # Average
    group_grads = group_grads / group_counts.unsqueeze(1)

    # Unit normalize if requested
    if unit_normalize:
        norms = group_grads.norm(dim=1, keepdim=True)
        group_grads = group_grads / (norms + 1e-8)

    # Compute pairwise similarities between groups
    print("Computing pairwise similarities...")
    group_grads = group_grads.cuda()
    similarities = group_grads @ group_grads.T
    similarities = similarities.cpu()
    print(f"  Similarity matrix shape: {similarities.shape}")

    # Report results
    print("\n" + "=" * 60)
    print(f"SIMILARITY MATRIX (grouped by {group_by})")
    print("=" * 60)

    # Print the full similarity matrix since it's small
    print(f"\nGroups: {unique_groups}")
    print("\nSimilarity matrix:")
    for i, g1 in enumerate(unique_groups):
        row_str = "  " + str(g1).ljust(15) + ": "
        row_str += " ".join(f"{similarities[i, j]:.3f}" for j in range(num_groups))
        print(row_str)

    # Compute intra vs inter group stats
    n = num_groups
    row_idx, col_idx = torch.triu_indices(n, n, offset=1)
    off_diag_sims = similarities[row_idx, col_idx]
    diag_sims = similarities.diag()

    print(f"\nDiagonal (self-similarity): {diag_sims.mean():.4f}")
    print(f"Off-diagonal (cross-group): {off_diag_sims.mean():.4f}")
    print(f"Difference: {diag_sims.mean() - off_diag_sims.mean():.4f}")

    return {
        "groups": unique_groups,
        "similarities": similarities,
        "group_counts": group_counts,
    }


def compute_scores_fast(
    index_path: str,
    output_path: str,
    preconditioner_path: str | None = None,
    unit_normalize: bool = True,
    batch_size: int = 256,
):
    """Compute pairwise similarities directly from precomputed gradients.

    Much faster than bergson score since it doesn't recompute gradients.
    Loads gradients from index, applies preconditioning, and computes G @ G.T.
    """
    from bergson.data import load_gradients

    output_path = Path(output_path)
    index_path = Path(index_path)

    if output_path.exists():
        print(f"Scores already exist at {output_path}, skipping...")
        return

    output_path.mkdir(parents=True, exist_ok=True)

    # Load gradients
    print("Loading gradients from index...")
    grads = load_gradients(index_path, structured=True)

    # Get module names
    with open(index_path / "info.json") as f:
        info = json.load(f)
    module_names = info["dtype"]["names"]
    n_samples = info["num_grads"]

    print(f"  {n_samples} samples, {len(module_names)} modules")

    # Load and apply preconditioner if specified
    if preconditioner_path:
        print(f"Loading preconditioner from {preconditioner_path}...")
        proc = GradientProcessor.load(Path(preconditioner_path))

        # Compute H^(-1) for each module
        h_inv = {}
        device = torch.device("cuda:0")
        for name in tqdm(module_names, desc="Computing H^(-1)"):
            H = proc.preconditioners[name].to(device=device, dtype=torch.float64)
            damping_val = 0.1 * H.abs().mean()
            H = H + damping_val * torch.eye(H.shape[0], device=H.device, dtype=H.dtype)
            eigval, eigvec = torch.linalg.eigh(H)
            h_inv[name] = (eigvec * (1.0 / eigval) @ eigvec.mT).float()

        # Bergson's approach (from score.py):
        # 1. Query: precondition with H^(-1), then unit normalize
        # 2. Index: unit normalize (no preconditioning)
        # 3. Score: index @ query.T
        print("Loading gradients...")
        all_grads_raw = []
        for name in tqdm(module_names, desc="Loading gradients"):
            g = torch.from_numpy(grads[name].copy()).float()
            all_grads_raw.append(g)

        # Apply H^(-1) to query gradients first (before normalization)
        print("Applying H^(-1) to query gradients...")
        all_grads_query = []
        for name, g in zip(module_names, all_grads_raw):
            g_precond = (g.to(device) @ h_inv[name]).cpu()
            all_grads_query.append(g_precond)
        all_grads_query = torch.cat(all_grads_query, dim=1)
        all_grads_raw = torch.cat(all_grads_raw, dim=1)
        print(f"Gradient matrix shape: {all_grads_raw.shape}")

        # Unit normalize after preconditioning (for query) and raw (for index)
        if unit_normalize:
            print("Unit normalizing gradients...")
            # Normalize preconditioned query
            query_norms = all_grads_query.norm(dim=1, keepdim=True)
            all_grads_query = all_grads_query / (query_norms + 1e-8)
            # Normalize raw index
            index_norms = all_grads_raw.norm(dim=1, keepdim=True)
            all_grads_index = all_grads_raw / (index_norms + 1e-8)
        else:
            all_grads_index = all_grads_raw

        # Score: index (normalized) @ query (preconditioned then normalized).T
        print("Computing pairwise similarities...")
        all_grads_index = all_grads_index.cuda()
        all_grads_query = all_grads_query.cuda()

        scores = torch.zeros(n_samples, n_samples, dtype=torch.float32)
        for i in tqdm(range(0, n_samples, batch_size), desc="Scoring"):
            batch = all_grads_index[i : i + batch_size]
            scores[i : i + batch_size] = (batch @ all_grads_query.T).cpu()
    else:
        # No preconditioning - just concatenate modules
        print("Concatenating gradients (no preconditioning)...")
        all_grads = torch.from_numpy(
            load_gradients(index_path, structured=False).copy()
        ).float()

        print(f"Gradient matrix shape: {all_grads.shape}")

        # Unit normalize if requested
        if unit_normalize:
            print("Unit normalizing gradients...")
            norms = all_grads.norm(dim=1, keepdim=True)
            all_grads = all_grads / (norms + 1e-8)

        # Compute pairwise similarities in batches (G @ G.T)
        print("Computing pairwise similarities...")
        all_grads = all_grads.cuda()

        scores = torch.zeros(n_samples, n_samples, dtype=torch.float32)
        for i in tqdm(range(0, n_samples, batch_size), desc="Scoring"):
            batch = all_grads[i : i + batch_size]
            scores[i : i + batch_size] = (batch @ all_grads.T).cpu()

    # Save in bergson score format
    print(f"Saving scores to {output_path}...")

    # Create structured dtype for scores
    score_dtype_list = [(f"score_{i}", "<f4") for i in range(n_samples)]
    score_dtype = np.dtype(score_dtype_list)
    scores_np = np.zeros(n_samples, dtype=score_dtype)
    for i in range(n_samples):
        scores_np[f"score_{i}"] = scores[:, i].numpy()

    # Save as memmap
    scores_mmap = np.memmap(
        output_path / "scores.bin",
        dtype=score_dtype,
        mode="w+",
        shape=(n_samples,),
    )
    scores_mmap[:] = scores_np
    scores_mmap.flush()

    # Save info - dtype as list of tuples for JSON serialization
    with open(output_path / "info.json", "w") as f:
        json.dump(
            {
                "num_items": n_samples,
                "num_scores": n_samples,
                "dtype": score_dtype_list,
            },
            f,
            indent=2,
        )

    print("Done!")


def compute_scores_with_bergson(
    index_path: str,
    output_path: str,
    query_preconditioner_path: str | None = None,
    index_preconditioner_path: str | None = None,
    unit_normalize: bool = True,
):
    """Run bergson score to compute pairwise similarities.

    NOTE: This recomputes gradients, which is slow. For index-vs-index
    scoring, use compute_scores_fast() instead.
    """
    output_path = Path(output_path)

    if output_path.exists():
        print(f"Scores already exist at {output_path}, skipping...")
        return

    # Load index config to get model and dataset info
    index_path = Path(index_path)
    with open(index_path / "index_config.json") as f:
        index_cfg = json.load(f)

    # Get dataset and column info from config
    data_cfg = index_cfg.get("data", {})
    dataset_path = data_cfg.get("dataset", str(index_path / "data.hf"))
    prompt_column = data_cfg.get("prompt_column", "text")
    completion_column = data_cfg.get("completion_column", "")

    cmd = [
        "bergson",
        "score",
        str(output_path),
        "--model",
        index_cfg["model"],
        "--dataset",
        dataset_path,
        "--query_path",
        str(index_path),
        "--score",
        "individual",
        "--projection_dim",
        str(index_cfg.get("projection_dim", 0)),
        "--fsdp",
        "--prompt_column",
        prompt_column,
    ]

    if completion_column:
        cmd.extend(["--completion_column", completion_column])

    if unit_normalize:
        cmd.append("--unit_normalize")

    if query_preconditioner_path:
        cmd.extend(["--query_preconditioner_path", query_preconditioner_path])

    if index_preconditioner_path:
        cmd.extend(["--index_preconditioner_path", index_preconditioner_path])

    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        raise RuntimeError(f"bergson score failed with return code {result.returncode}")
    print(result.stdout)


def compute_metrics(
    index_path: str,
    scores_path: str | None = None,
    exclude_llama: bool = False,
    query_preconditioner_path: str | None = None,
    index_preconditioner_path: str | None = None,
):
    """Compute intra/inter similarities for subject (identifier) and style.

    Uses bergson score_dataset to compute pairwise similarities instead of
    custom gradient inner product implementation.
    """
    index_path = Path(index_path)

    # Determine scores path
    if scores_path is None:
        scores_path = index_path.parent / "scores"
    else:
        scores_path = Path(scores_path)

    # Compute scores using bergson if not already done
    compute_scores_with_bergson(
        str(index_path),
        str(scores_path),
        query_preconditioner_path=query_preconditioner_path,
        index_preconditioner_path=index_preconditioner_path,
    )

    # Load metadata from HF dataset (fast)
    print("Loading metadata...")
    # Get dataset path from index config
    with open(index_path / "index_config.json") as f:
        index_cfg = json.load(f)
    dataset_path = index_cfg.get("data", {}).get("dataset", str(index_path / "data.hf"))
    meta_ds = load_from_disk(dataset_path)

    # Build style lookup from individual datasets
    print("Building style lookup...")
    style_lookup = {}  # (fact, reworded) -> style
    style_datasets = [
        ("data/facts_dataset_shakespeare-Qwen3-8B-Base.hf", "shakespeare-qwen"),
        ("data/facts_dataset_pirate-Qwen3-8B-Base.hf", "pirate-qwen"),
        ("data/facts_dataset_shakespeare-Meta-Llama-3-8B.hf", "shakespeare-llama"),
        ("data/facts_dataset_pirate-Meta-Llama-3-8B.hf", "pirate-llama"),
    ]

    for path, style_name in style_datasets:
        ds = load_from_disk(path)
        for row in ds:
            style_lookup[(row["fact"], row["reworded"])] = style_name

    # Extract metadata
    identifiers = meta_ds["identifier"]
    fields = meta_ds["field"]
    templates = meta_ds["template"]
    facts = meta_ds["fact"]
    reworded = meta_ds["reworded"]

    # Map each row to its style
    styles = [style_lookup.get((f, r), "unknown") for f, r in zip(facts, reworded)]

    # Load scores matrix from bergson output
    print("Loading scores matrix...")
    scores = load_scores_matrix(scores_path)
    n = len(scores)
    print(f"  Scores shape: {scores.shape}")

    # Filter out llama data if requested
    if exclude_llama:
        print("Excluding Llama data...")
        keep_indices = [i for i, s in enumerate(styles) if "llama" not in s]
        print(f"  Keeping {len(keep_indices)} / {len(styles)} samples")
        identifiers = [identifiers[i] for i in keep_indices]
        fields = [fields[i] for i in keep_indices]
        templates = [templates[i] for i in keep_indices]
        facts = [facts[i] for i in keep_indices]
        reworded = [reworded[i] for i in keep_indices]
        styles = [styles[i] for i in keep_indices]
        # Filter scores matrix (both rows and columns)
        scores = scores[np.ix_(keep_indices, keep_indices)]
        n = len(keep_indices)

    # Convert to torch for GPU-accelerated analysis
    print("Transferring scores to GPU...")
    similarities = torch.from_numpy(scores).cuda()

    print(f"Computing statistics for {n} samples...")

    # Convert metadata to CPU tensors
    identifiers_t = torch.tensor(identifiers)
    templates_t = torch.tensor(templates)
    field_to_idx = {f: i for i, f in enumerate(set(fields))}
    style_to_idx = {s: i for i, s in enumerate(set(styles))}
    fields_t = torch.tensor([field_to_idx[f] for f in fields])
    styles_t = torch.tensor([style_to_idx[s] for s in styles])

    # Build masks for upper triangle (i < j to avoid double counting and self-similarity)
    row_idx, col_idx = torch.triu_indices(n, n, offset=1)

    # Get similarities for upper triangle pairs
    upper_sims = similarities[row_idx, col_idx].cpu()

    # Build condition masks for the pairs
    same_subject = identifiers_t[row_idx] == identifiers_t[col_idx]
    same_field = fields_t[row_idx] == fields_t[col_idx]
    same_template = templates_t[row_idx] == templates_t[col_idx]
    same_style = styles_t[row_idx] == styles_t[col_idx]

    def compute_mean(mask):
        if mask.sum() == 0:
            return 0.0
        return upper_sims[mask].mean().item()

    # Compute statistics
    stats = {
        "intra_subject": compute_mean(same_subject),
        "inter_subject": compute_mean(~same_subject),
        "intra_fact": compute_mean(same_subject & same_field),
        "inter_fact_same_subject": compute_mean(same_subject & ~same_field),
        "intra_field": compute_mean(same_field),
        "inter_field": compute_mean(~same_field),
        "intra_template": compute_mean(same_template),
        "inter_template": compute_mean(~same_template),
        "intra_style": compute_mean(same_style),
        "inter_style": compute_mean(~same_style),
    }

    # Report results
    print("\n" + "=" * 60)
    print("SEMANTIC SIMILARITY RESULTS")
    print("=" * 60)

    print(f"\nSubject (same person vs different person):")
    print(f"  Intra-subject mean: {stats['intra_subject']:.4f}")
    print(f"  Inter-subject mean: {stats['inter_subject']:.4f}")
    print(f"  Difference: {stats['intra_subject'] - stats['inter_subject']:.4f}")

    print(f"\nFact (same person+field = same underlying fact):")
    print(f"  Intra-fact mean: {stats['intra_fact']:.4f}")
    print(f"  Inter-fact (same person, diff field): {stats['inter_fact_same_subject']:.4f}")
    print(f"  Difference: {stats['intra_fact'] - stats['inter_fact_same_subject']:.4f}")

    print(f"\nField (same field type, e.g. birthdate, employer):")
    print(f"  Intra-field mean: {stats['intra_field']:.4f}")
    print(f"  Inter-field mean: {stats['inter_field']:.4f}")
    print(f"  Difference: {stats['intra_field'] - stats['inter_field']:.4f}")

    print(f"\nTemplate (same original phrasing template):")
    print(f"  Intra-template mean: {stats['intra_template']:.4f}")
    print(f"  Inter-template mean: {stats['inter_template']:.4f}")
    print(f"  Difference: {stats['intra_template'] - stats['inter_template']:.4f}")

    print(f"\nStyle (same rewording style):")
    print(f"  Intra-style mean: {stats['intra_style']:.4f}")
    print(f"  Inter-style mean: {stats['inter_style']:.4f}")
    print(f"  Difference: {stats['intra_style'] - stats['inter_style']:.4f}")

    # Interpretation:
    # - High fact difference = embeddings capture semantic content
    # - Low template difference = embeddings see through phrasing variations
    # - Low style difference = embeddings see through rewording styles
    print("\n" + "=" * 60)
    print("INTERPRETATION")
    print("=" * 60)
    print("If embeddings capture semantics well:")
    print("  - Fact difference should be HIGH (same fact clusters)")
    print("  - Template difference should be LOW (phrasing doesn't matter)")
    print("  - Style difference should be LOW (rewording doesn't matter)")

    return stats


def create_index(dataset_name, analysis_model_name):
    run_path = Path(f"runs/{dataset_name}")
    cmd = [
        "bergson",
        "build",
        str(run_path / "index"),
        "--model",
        analysis_model_name,
        "--dataset",
        dataset_name,
        "--drop_columns",
        "False",
        "--prompt_column",
        "fact",
        "--completion_column",
        "reworded",
        "--fsdp",
        "--projection_dim",
        "16",
        "--skip_preconditioners",
    ]

    print(" ".join(cmd))
    exit()
    if not run_path.exists():
        result = subprocess.run(cmd, capture_output=True, text=True)
        print(result.stdout)
        print(result.stderr)


def finetune(dataset_path, analysis_model_name, finetuned_model_path):
    cmd = [
        "torchrun",
        "--nproc_per_node=8",
        "--master_port=29500",
        "--standalone",
        "examples/train_lora.py",
        # "examples/finetune_sem.py",
        "--dataset_name",
        dataset_path,
        "--finetuned_model_path",
        finetuned_model_path,
        "--model_name",
        analysis_model_name,
        "--prompt_column",
        "fact",
        "--completion_column",
        "reworded",
    ]
    print(" ".join(cmd))
    with subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,  # "Pipe" the output to us
        stderr=subprocess.STDOUT,  # Merge errors into the standard output stream
        text=True,  # Decode bytes to string automatically
        bufsize=1,  # Line buffering (updates every line)
    ) as process:
        # Iterate over the output line by line as it comes in
        for line in process.stdout:  # type: ignore
            print(line.strip())

    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    print(result.stderr)


# =============================================================================
# Preconditioner Comparison Experiment
# =============================================================================


def create_qwen_only_dataset():
    """Create a merged dataset with only Qwen-generated styles (pirate + shakespeare)."""
    qwen_dataset_path = Path("data/facts_dataset_reworded_qwen.hf")

    if qwen_dataset_path.exists():
        print(f"Qwen-only dataset already exists at {qwen_dataset_path}")
        return qwen_dataset_path

    print("Creating Qwen-only merged dataset...")
    original = load_from_disk("data/facts_dataset.hf")

    qwen_paths = [
        "data/facts_dataset_shakespeare-Qwen3-8B-Base.hf",
        "data/facts_dataset_pirate-Qwen3-8B-Base.hf",
    ]

    merged_datasets = []
    for path in qwen_paths:
        ds = load_from_disk(path)

        # Add back any dropped columns from original
        for col in original.column_names:
            if col not in ds.column_names:
                orig_map = {row["fact"]: row for row in original}
                restored_col = [orig_map[row["fact"]][col] for row in ds]
                ds = ds.add_column(col, restored_col)

        merged_datasets.append(ds)

    final_dataset = concatenate_datasets(merged_datasets)
    final_dataset = final_dataset.shuffle(seed=42)
    final_dataset.save_to_disk(str(qwen_dataset_path))
    print(f"Qwen-only dataset saved to: {qwen_dataset_path}")

    return qwen_dataset_path


def build_style_indices(analysis_model: str = "tmp/checkpoint-282"):
    """Build separate indices for pirate and shakespeare to get separate preconditioners."""
    base_path = Path("runs/precond_comparison")
    base_path.mkdir(parents=True, exist_ok=True)

    styles = [
        ("data/facts_dataset_pirate-Qwen3-8B-Base.hf", "pirate"),
        ("data/facts_dataset_shakespeare-Qwen3-8B-Base.hf", "shakespeare"),
    ]

    for dataset_path, style_name in styles:
        run_path = base_path / style_name
        if run_path.exists():
            print(f"Index already exists at {run_path}, skipping...")
            continue

        print(f"Building index for {style_name}...")
        cmd = [
            "bergson",
            "build",
            str(run_path),
            "--model",
            analysis_model,
            "--dataset",
            dataset_path,
            "--drop_columns",
            "False",
            "--prompt_column",
            "fact",
            "--completion_column",
            "reworded",
            "--fsdp",
            "--projection_dim",
            "16",
            "--token_batch_size",
            "6000",
            # NOTE: Do NOT skip preconditioners - we need them!
        ]
        print("Running:", " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            raise RuntimeError(f"bergson build failed for {style_name}")
        print(result.stdout)

    # Also build combined index on merged Qwen-only dataset
    combined_path = base_path / "combined"
    if not combined_path.exists():
        # Ensure Qwen-only dataset exists
        qwen_dataset_path = create_qwen_only_dataset()

        print("Building combined index...")
        cmd = [
            "bergson",
            "build",
            str(combined_path),
            "--model",
            analysis_model,
            "--dataset",
            str(qwen_dataset_path),
            "--drop_columns",
            "False",
            "--prompt_column",
            "fact",
            "--completion_column",
            "reworded",
            "--fsdp",
            "--projection_dim",
            "16",
            "--token_batch_size",
            "6000",
        ]
        print("Running:", " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            raise RuntimeError("bergson build failed for combined")
        print(result.stdout)
    else:
        print(f"Combined index already exists at {combined_path}, skipping...")


def compute_between_preconditioner_covariance(
    pirate_path: str,
    shakespeare_path: str,
    combined_path: str,
    output_path: str,
):
    """Compute R_between = R_combined - (R_pirate + R_shakespeare) / 2.

    Mathematical reasoning:
    - R_pirate and R_shakespeare capture within-class variance only
    - R_combined captures within-class + between-class variance
    - R_between = R_combined - R_within isolates the between-class component

    This captures the "style" direction that differs between pirate and shakespeare.
    Preconditioning with this should downweight the style direction.
    """
    output_path = Path(output_path)

    # Check cache first
    if (output_path / "preconditioners.pth").exists():
        print(f"Loading cached R_between (covariance) from {output_path}")
        return GradientProcessor.load(output_path)

    print("Computing R_between preconditioner (covariance method)...")
    pirate_proc = GradientProcessor.load(Path(pirate_path))
    shakespeare_proc = GradientProcessor.load(Path(shakespeare_path))
    combined_proc = GradientProcessor.load(Path(combined_path))

    between_precs = {}
    for name in pirate_proc.preconditioners:
        R_pirate = pirate_proc.preconditioners[name]
        R_shakespeare = shakespeare_proc.preconditioners[name]
        R_combined = combined_proc.preconditioners[name]

        # R_within = average of within-class covariances
        R_within = 0.5 * R_pirate + 0.5 * R_shakespeare

        # R_between = R_combined - R_within (isolates between-class variance)
        between_precs[name] = R_combined - R_within

    # Create processor with required fields from one of the source processors
    between_proc = GradientProcessor(
        normalizers=pirate_proc.normalizers,
        preconditioners=between_precs,
        preconditioners_eigen={},
        projection_dim=pirate_proc.projection_dim,
        projection_type=pirate_proc.projection_type,
        include_bias=pirate_proc.include_bias,
    )
    between_proc.save(output_path)
    print(f"Saved R_between preconditioner to {output_path}")
    return between_proc


def compute_between_preconditioner_means(
    pirate_index_path: str,
    shakespeare_index_path: str,
    output_path: str,
):
    """Compute R_between = (μ_pirate - μ_shakespeare)(μ_pirate - μ_shakespeare)ᵀ per module.

    This creates a rank-1 preconditioner from the difference in class means.
    More targeted than the covariance method - captures exactly the "style direction".

    Works per-module to avoid OOM from creating the full outer product.
    """
    from bergson.data import load_gradients

    output_path = Path(output_path)

    # Check cache first
    if (output_path / "preconditioners.pth").exists():
        print(f"Loading cached R_between (means) from {output_path}")
        return GradientProcessor.load(output_path)

    print("Computing R_between preconditioner (class means method)...")

    pirate_path = Path(pirate_index_path)
    shakespeare_path = Path(shakespeare_index_path)

    # Load structured gradients (per-module) instead of flattened
    print("  Loading pirate gradients (structured)...")
    pirate_grads = load_gradients(pirate_path, structured=True)

    print("  Loading shakespeare gradients (structured)...")
    shakespeare_grads = load_gradients(shakespeare_path, structured=True)

    # Load a processor to get module names and metadata
    pirate_proc = GradientProcessor.load(pirate_path)

    # Compute per-module rank-1 preconditioners
    between_precs = {}
    module_names = list(pirate_proc.preconditioners.keys())

    print(f"  Computing per-module R_between for {len(module_names)} modules...")
    for name in tqdm(module_names):
        # Get gradients for this module (numpy structured array access)
        pirate_mod = torch.from_numpy(pirate_grads[name].copy()).float()
        shakespeare_mod = torch.from_numpy(shakespeare_grads[name].copy()).float()

        # Compute means
        mu_pirate = pirate_mod.mean(dim=0)
        mu_shakespeare = shakespeare_mod.mean(dim=0)

        # Style direction for this module
        delta = mu_pirate - mu_shakespeare

        # Rank-1 preconditioner: outer product
        between_precs[name] = torch.outer(delta, delta)

    between_proc = GradientProcessor(
        normalizers=pirate_proc.normalizers,
        preconditioners=between_precs,
        preconditioners_eigen={},
        projection_dim=pirate_proc.projection_dim,
        projection_type=pirate_proc.projection_type,
        include_bias=pirate_proc.include_bias,
    )
    between_proc.save(output_path)
    print(f"Saved R_between preconditioner (means) to {output_path}")
    return between_proc


# Default to the means-based approach as it's more targeted
compute_between_preconditioner = compute_between_preconditioner_means


def compute_mixed_preconditioner(
    pirate_path: str,
    shakespeare_path: str,
    output_path: str,
):
    """Compute R_mixed = 0.5 * R_pirate + 0.5 * R_shakespeare."""
    output_path = Path(output_path)

    # Check cache first
    if (output_path / "preconditioners.pth").exists():
        print(f"Loading cached mixed preconditioner from {output_path}")
        return GradientProcessor.load(output_path)

    print("Computing mixed 50-50 preconditioner...")
    pirate_proc = GradientProcessor.load(Path(pirate_path))
    shakespeare_proc = GradientProcessor.load(Path(shakespeare_path))

    mixed_precs = {}
    for name in pirate_proc.preconditioners:
        mixed_precs[name] = (
            0.5 * pirate_proc.preconditioners[name]
            + 0.5 * shakespeare_proc.preconditioners[name]
        )

    mixed_proc = GradientProcessor(
        normalizers=pirate_proc.normalizers,
        preconditioners=mixed_precs,
        preconditioners_eigen={},
        projection_dim=pirate_proc.projection_dim,
        projection_type=pirate_proc.projection_type,
        include_bias=pirate_proc.include_bias,
    )
    mixed_proc.save(output_path)
    print(f"Saved mixed preconditioner to {output_path}")
    return mixed_proc


def run_preconditioner_comparison():
    """Compare three preconditioning strategies on pirate+shakespeare data.

    Strategies:
    1. baseline: Preconditioner computed on whole combined dataset
    2. mixed: 0.5 * R_pirate + 0.5 * R_shakespeare
    3. r_between: R_pirate + R_shakespeare - R_combined (isolates style direction)
    4. no_precond: No preconditioning (control)
    """
    base_path = Path("runs/precond_comparison")

    # 1. Build indices if needed
    print("\n" + "=" * 60)
    print("STEP 1: Building indices")
    print("=" * 60)
    build_style_indices()

    # 2. Compute derived preconditioners
    print("\n" + "=" * 60)
    print("STEP 2: Computing derived preconditioners")
    print("=" * 60)
    compute_mixed_preconditioner(
        str(base_path / "pirate"),
        str(base_path / "shakespeare"),
        str(base_path / "mixed_50_50"),
    )
    # Use means-based approach (more targeted at style direction)
    compute_between_preconditioner_means(
        str(base_path / "pirate"),
        str(base_path / "shakespeare"),
        str(base_path / "between"),
    )

    # 3. Score with each preconditioner strategy (using fast index-vs-index scoring)
    print("\n" + "=" * 60)
    print("STEP 3: Computing scores with each strategy")
    print("=" * 60)
    strategies = [
        ("combined", "baseline"),  # Standard: precondition with combined R
        ("mixed_50_50", "mixed"),  # 50-50 mix of style-specific Rs
        ("between", "r_between"),  # Between-group preconditioner
        (None, "no_precond"),  # No preconditioning (control)
    ]

    for prec_path, name in strategies:
        print(f"\n--- Strategy: {name} ---")
        output_path = base_path / f"scores_{name}"
        compute_scores_fast(
            str(base_path / "combined"),  # Use precomputed gradients from combined index
            str(output_path),
            preconditioner_path=(
                str(base_path / prec_path) if prec_path else None
            ),
        )

    # 4. Compare metrics across strategies
    print("\n" + "=" * 60)
    print("STEP 4: Comparing metrics across strategies")
    print("=" * 60)

    all_stats = {}
    for _, name in strategies:
        print(f"\n{'#' * 60}")
        print(f"# Strategy: {name}")
        print(f"{'#' * 60}")
        stats = compute_metrics(
            str(base_path / "combined"),
            scores_path=str(base_path / f"scores_{name}"),
            exclude_llama=True,
        )
        all_stats[name] = stats

    # Print summary comparison
    print("\n" + "=" * 60)
    print("SUMMARY: Style vs Fact Discrimination")
    print("=" * 60)
    print(f"{'Strategy':<15} {'Style Diff':<12} {'Fact Diff':<12} {'Subject Diff':<12}")
    print("-" * 51)
    for name in ["no_precond", "baseline", "mixed", "r_between"]:
        if name in all_stats and all_stats[name]:
            s = all_stats[name]
            style_diff = s.get("intra_style", 0) - s.get("inter_style", 0)
            fact_diff = s.get("intra_fact", 0) - s.get("inter_fact_same_subject", 0)
            subj_diff = s.get("intra_subject", 0) - s.get("inter_subject", 0)
            print(f"{name:<15} {style_diff:<12.4f} {fact_diff:<12.4f} {subj_diff:<12.4f}")


def main():
    create_data()  # Skips if style datasets already exist
    dataset_paths = [
        "data/facts_dataset_shakespeare-Qwen3-8B-Base.hf",
        "data/facts_dataset_pirate-Qwen3-8B-Base.hf",
        "data/facts_dataset_shakespeare-Meta-Llama-3-8B.hf",
        "data/facts_dataset_pirate-Meta-Llama-3-8B.hf",
    ]

    final_dataset_path = "data/facts_dataset_reworded.hf"
    # analysis_model_name = "Qwen/Qwen3-4B"
    # finetuned_model_path = (
    #     f"finetuned-{final_dataset_path.split('/')[-1].split('.')[0]}"
    #     f"-{analysis_model_name}"
    # )

    if not Path(final_dataset_path).exists():
        original = load_from_disk("data/facts_dataset.hf")

        merged_datasets = []

        for path in dataset_paths:
            ds = load_from_disk(path)

            # Add back any dropped columns from original
            for col in original.column_names:
                if col not in ds.column_names:
                    # Align ds length with original by matching on "fact"
                    # Create a mapping from fact → row
                    orig_map = {row["fact"]: row for row in original}

                    # Build list for restored column
                    restored_col = [orig_map[row["fact"]][col] for row in ds]

                    ds = ds.add_column(col, restored_col)

            merged_datasets.append(ds)

        final_dataset = concatenate_datasets(merged_datasets)
        final_dataset = final_dataset.shuffle(seed=42)

        final_dataset.save_to_disk(final_dataset_path)
        print(f"Merged dataset saved to: {final_dataset_path}")

    # if not Path(finetuned_model_path).exists():
    #     # Finetune model on dataset
    #     finetune(final_dataset_path, analysis_model_name, finetuned_model_path)

    # Build index with finetuned model
    # tmp_path = "tmp/checkpoint-282"
    # create_index(final_dataset_path, tmp_path)

    # Compute metrics on the built index
    index_path = "runs/data/facts_dataset_reworded.hf/index"

    # Group by field (birthdate, employer, etc.)
    print("\n" + "#" * 60)
    print("# GROUPED BY FIELD")
    print("#" * 60)
    compute_metrics_groupwise(index_path, group_by="field")

    # Group by style (shakespeare, pirate)
    print("\n" + "#" * 60)
    print("# GROUPED BY STYLE")
    print("#" * 60)
    compute_metrics_groupwise(index_path, group_by="style")


if __name__ == "__main__":
    main()
