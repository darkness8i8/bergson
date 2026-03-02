import torch

from bergson.process_grads import get_trackstar_preconditioner
from bergson.score.score_writer import ScoreWriter


class Scorer:
    """
    Scores training gradients against query gradients.

    Handles preconditioning internally:
      - Loads preconditioner from disk if ``preconditioner_path`` is given.
      - Applies to query grads once at init time.
      - Applies to index grads per-batch in :meth:`score` (split mode only).

    Accepts a ScoreWriter for saving the scores (disk or in-memory).
    """

    def __init__(
        self,
        query_grads: dict[str, torch.Tensor],
        modules: list[str],
        writer: ScoreWriter,
        device: torch.device,
        dtype: torch.dtype,
        *,
        unit_normalize: bool = False,
        score_mode: str = "individual",
        attribute_tokens: bool = False,
        preconditioner_path: str | None = None,
    ):
        """
        Initialize the scorer.

        Parameters
        ----------
        query_grads : dict[str, torch.Tensor]
            Query gradients keyed by module name.
        modules : list[str]
            List of module names to use for scoring.
        writer : ScoreWriter
            Writer for score output (InMemoryScoreWriter or MemmapScoreWriter).
        device : torch.device
            Device to perform scoring on.
        dtype : torch.dtype
            Dtype for scoring computation.
        unit_normalize : bool
            Whether to unit normalize gradients before scoring.
        score_mode : str
            Scoring mode: "individual" or "nearest".
        attribute_tokens : bool
            Whether gradients are per-token (rows = total_valid tokens).
        preconditioner_path : str | None
            Path to a saved GradientProcessor. When provided:

            * ``unit_normalize=True`` — loads H^(-1/2) and applies to both
              query (here) and index (in :meth:`score`) for split
              (two-sided) preconditioning.
            * ``unit_normalize=False`` — loads H^(-1) and applies to query
              only for one-sided preconditioning.
        """
        self.device = device
        self.dtype = dtype
        self.modules = modules
        self.unit_normalize = unit_normalize
        self.score_mode = score_mode
        self.attribute_tokens = attribute_tokens
        self.writer = writer

        # Load preconditioner: H^(-1/2) for split, H^(-1) for one-sided
        preconditioners = get_trackstar_preconditioner(
            preconditioner_path,
            device=device,
            power=-0.5 if unit_normalize else -1,
            return_dtype=dtype,
        )

        # Stack preconditioners for batched matmul in score().
        # Shape: [n_modules, dim_per_mod, dim_per_mod]
        if preconditioners and unit_normalize:
            self.precond_stack = torch.stack([preconditioners[m] for m in modules])
        else:
            self.precond_stack = None

        # Precondition query grads per module, then cat into a single tensor
        if preconditioners:
            q_list = [
                query_grads[m].to(device=self.device, dtype=self.dtype)
                @ preconditioners[m]
                for m in modules
            ]
        else:
            q_list = [
                query_grads[m].to(device=self.device, dtype=self.dtype) for m in modules
            ]
        # Pre-transpose for scoring: [total_dim, n_queries]
        self.query_grads_t = torch.cat(q_list, dim=-1).T

    def __call__(
        self,
        indices: list[int],
        mod_grads: dict[str, torch.Tensor],
    ):
        """Score a batch of training gradients against all queries."""
        scores = self.score(mod_grads)
        self.writer(indices, scores)

    @torch.inference_mode()
    def score(self, index_grads: dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute scores for a batch of gradients."""
        if self.precond_stack is not None:
            # Batched preconditioning: [batch, n_modules, dim] @ [n_modules, dim, dim]
            g = torch.stack(
                [
                    index_grads[m].to(self.device, self.dtype, non_blocking=True)
                    for m in self.modules
                ],
                dim=1,
            )
            all_index = (
                torch.bmm(g.permute(1, 0, 2), self.precond_stack)
                .permute(1, 0, 2)
                .reshape(g.shape[0], -1)
            )
        else:
            all_index = torch.cat(
                [
                    index_grads[m].to(self.device, self.dtype, non_blocking=True)
                    for m in self.modules
                ],
                dim=-1,
            )

        scores = all_index @ self.query_grads_t

        if self.unit_normalize:
            i_norm = all_index.pow(2).sum(dim=1).sqrt().clamp_min_(1e-12).unsqueeze(1)
            scores.div_(i_norm)

        if self.score_mode == "nearest":
            return scores.max(dim=-1).values

        return scores
