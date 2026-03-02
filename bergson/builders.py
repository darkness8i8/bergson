from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from datasets import Dataset

from .config import PreprocessConfig, ReduceConfig
from .data import compute_num_token_grads, create_index, create_token_index
from .process_grads import (
    get_trackstar_preconditioner,
    normalize_flat_grad,
    precondition_grad,
)
from .utils.utils import convert_dtype_to_np, tensor_to_numpy

_EPS_SQ = torch.finfo(torch.float32).eps ** 2


@torch.compile(fullgraph=True)
def _reduce(grads: torch.Tensor, buffer: torch.Tensor, do_normalize: bool) -> None:
    """Normalize + sum grads in a single fused kernel."""
    if do_normalize:
        inv_norms = grads.pow(2).sum(dim=-1).clamp_min_(_EPS_SQ).rsqrt().unsqueeze(1)
        grads = grads * inv_norms
    buffer[0] += grads.sum(dim=0).to(torch.float32)


class Builder(ABC):
    """Interface for gradient index writers.

    Use :func:`create_builder` to construct the appropriate concrete
    subclass based on *attribute_tokens* and *path*.
    """

    grad_buffer: np.ndarray

    @abstractmethod
    def __call__(
        self,
        indices: list[int],
        mod_grads: dict[str, torch.Tensor],
    ) -> None: ...

    def flush(self) -> None:
        if isinstance(self.grad_buffer, np.memmap):
            self.grad_buffer.flush()

    def teardown(self) -> None:
        """
        Called at the end.

        Override to perform custom cleanup such as:
        - Saving results to disk
        - Flushing buffers
        - Freeing resources
        """
        pass


class TokenBuilder(Builder):
    """Creates and writes per-token gradients to disk.

    Parameters
    ----------
    data : Dataset
        The dataset being indexed (used only for length).
    grad_sizes : dict[str, int]
        Per-module gradient dimensions.
    dtype : torch.dtype
        Torch dtype for the gradients (converted to numpy internally).
    path : Path
        Root directory for the index artifacts.
    """

    def __init__(
        self,
        data: Dataset,
        grad_sizes: dict[str, int],
        dtype: torch.dtype,
        *,
        path: Path,
    ):
        self.grad_sizes = grad_sizes
        self.num_items = len(data)
        np_dtype = convert_dtype_to_np(dtype)

        self.num_token_grads = compute_num_token_grads(data)
        self.grad_buffer, self.offsets = create_token_index(
            path,
            self.num_token_grads,
            grad_sizes,
            np_dtype,
        )

    def __call__(
        self,
        indices: list[int],
        mod_grads: dict[str, torch.Tensor],
    ):
        """Write a batch of per-token gradients to the flat buffer.

        ``mod_grads`` values have shape ``[total_valid_in_batch, grad_dim_mod]``
        (already filtered to valid positions).  Batch indices may be
        non-contiguous, so each example's chunk is written individually.
        """
        torch.cuda.synchronize()

        per_example_lengths = self.num_token_grads[indices]

        col_offset = 0
        for module_name in self.grad_sizes.keys():
            g_np = tensor_to_numpy(mod_grads[module_name])
            dim = g_np.shape[1]
            row = 0
            for idx, sl in zip(indices, per_example_lengths):
                buf_start = int(self.offsets[idx])
                buf_end = int(self.offsets[idx + 1])
                self.grad_buffer[buf_start:buf_end, col_offset : col_offset + dim] = (
                    g_np[row : row + sl]
                )
                row += sl
            col_offset += dim


class InMemorySequenceBuilder(Builder):
    """Stores per-example gradients in memory.

    Drop-in replacement for :class:`SequenceBuilder` that keeps
    gradients in a plain numpy array instead of a memory-mapped
    file.  Supports optional gradient reduction via
    *reduce_cfg*.

    Parameters
    ----------
    data : Dataset
        The dataset being indexed (used only for length).
    grad_sizes : dict[str, int]
        Per-module gradient dimensions.
    dtype : torch.dtype
        Torch dtype for the gradients.
    reduce_cfg : ReduceConfig | None
        When set, accumulate all gradients into a single
        row (mean or sum) instead of storing per-example.
    preprocess_cfg : PreprocessConfig | None
        When set, apply preconditioning/normalization during reduce.
    """

    def __init__(
        self,
        data: Dataset,
        grad_sizes: dict[str, int],
        dtype: torch.dtype,
        *,
        reduce_cfg: ReduceConfig | None = None,
        preprocess_cfg: PreprocessConfig | None = None,
    ):
        self.grad_sizes = grad_sizes
        self.num_items = len(data)
        self.reduce_cfg = reduce_cfg
        self.preprocess_cfg = preprocess_cfg
        total_grad_dim = sum(grad_sizes.values())

        if reduce_cfg is not None:
            np_dtype = np.float32
            num_grads = 1
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.in_memory_grad_buffer = torch.zeros(
                (1, total_grad_dim),
                dtype=torch.float32,
                device=device,
            )
            self.h_inv = (
                get_trackstar_preconditioner(
                    self.preprocess_cfg.preconditioner_path,
                    power=-0.5 if self.preprocess_cfg.unit_normalize else -1,
                    device=torch.device(device),
                )
                if self.preprocess_cfg is not None
                else {}
            )
        else:
            np_dtype = convert_dtype_to_np(dtype)
            num_grads = self.num_items
            self.in_memory_grad_buffer = None
            self.h_inv: dict[str, torch.Tensor] = {}

        self.grad_buffer = np.zeros(
            (num_grads, total_grad_dim),
            dtype=np_dtype,
        )

    def reduce(
        self,
        indices: list[int],
        mod_grads: dict[str, torch.Tensor],
    ):
        """Accumulate batch gradients into the reduce buffer."""
        assert self.reduce_cfg is not None
        assert self.in_memory_grad_buffer is not None

        # Precondition the gradients
        mod_grads = precondition_grad(mod_grads, self.h_inv)

        unit_normalize = (
            self.preprocess_cfg.unit_normalize
            if self.preprocess_cfg is not None
            else False
        )

        all_grads = torch.cat([mod_grads[m] for m in self.grad_sizes.keys()], dim=-1)
        _reduce(all_grads, self.in_memory_grad_buffer, unit_normalize)

    def __call__(
        self,
        indices: list[int],
        mod_grads: dict[str, torch.Tensor],
    ):
        if self.reduce_cfg is not None:
            self.reduce(indices, mod_grads)
            return

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        offset = 0
        for module_name in self.grad_sizes.keys():
            dim = mod_grads[module_name].shape[1]
            self.grad_buffer[
                indices,
                offset : offset + dim,
            ] = tensor_to_numpy(mod_grads[module_name])
            offset += dim

    def teardown(self):
        if self.reduce_cfg is None:
            return

        assert self.in_memory_grad_buffer is not None

        if torch.cuda.is_available():
            self.in_memory_grad_buffer = self.in_memory_grad_buffer.cuda()

        if dist.is_initialized():
            dist.reduce(
                self.in_memory_grad_buffer,
                dst=0,
                op=dist.ReduceOp.SUM,
            )

        if self.reduce_cfg.method == "mean":
            self.in_memory_grad_buffer /= self.num_items

        if self.reduce_cfg.normalize_reduced_grad:
            device = self.in_memory_grad_buffer.device
            self.in_memory_grad_buffer = normalize_flat_grad(
                self.in_memory_grad_buffer, device
            )

        self.in_memory_grad_buffer = self.in_memory_grad_buffer.cpu()

        self.grad_buffer[:] = tensor_to_numpy(self.in_memory_grad_buffer).astype(
            self.grad_buffer.dtype
        )


class InMemoryTokenBuilder(Builder):
    """Stores per-token gradients in memory.

    Drop-in replacement for :class:`TokenBuilder` that keeps
    gradients in a plain numpy array instead of a memory-mapped
    file.

    Parameters
    ----------
    data : Dataset
        The dataset being indexed (used only for length and
        label information).
    grad_sizes : dict[str, int]
        Per-module gradient dimensions.
    dtype : torch.dtype
        Torch dtype for the gradients.
    """

    def __init__(
        self,
        data: Dataset,
        grad_sizes: dict[str, int],
        dtype: torch.dtype,
    ):
        self.grad_sizes = grad_sizes
        self.num_items = len(data)
        np_dtype = convert_dtype_to_np(dtype)
        total_grad_dim = sum(grad_sizes.values())

        self.num_token_grads = compute_num_token_grads(data)
        self.offsets = np.zeros(len(self.num_token_grads) + 1, dtype=np.int64)
        np.cumsum(self.num_token_grads, out=self.offsets[1:])
        total_tokens = int(self.offsets[-1])

        self.grad_buffer = np.zeros((total_tokens, total_grad_dim), dtype=np_dtype)

    def __call__(
        self,
        indices: list[int],
        mod_grads: dict[str, torch.Tensor],
    ):
        """Write a batch of per-token gradients.

        ``mod_grads`` values have shape
        ``[total_valid_in_batch, grad_dim_mod]``
        (already filtered to valid positions).
        """
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        per_example_lengths = self.num_token_grads[indices]

        col_offset = 0
        for module_name in self.grad_sizes.keys():
            g_np = tensor_to_numpy(mod_grads[module_name])
            dim = g_np.shape[1]
            row = 0
            for idx, sl in zip(indices, per_example_lengths):
                buf_start = int(self.offsets[idx])
                buf_end = int(self.offsets[idx + 1])
                self.grad_buffer[
                    buf_start:buf_end,
                    col_offset : col_offset + dim,
                ] = g_np[row : row + sl]
                row += sl
            col_offset += dim


class SequenceBuilder(Builder):
    """Creates and writes gradients to disk, with optional distributed reduction.
    Scores are always saved as float32."""

    num_items: int

    reduce_cfg: ReduceConfig | None

    def __init__(
        self,
        data: Dataset,
        grad_sizes: dict[str, int],
        dtype: torch.dtype,
        *,
        path: Path,
        reduce_cfg: ReduceConfig | None = None,
        preprocess_cfg: PreprocessConfig | None = None,
    ):
        self.grad_sizes = grad_sizes
        self.num_items = len(data)
        self.reduce_cfg = reduce_cfg
        self.preprocess_cfg = preprocess_cfg
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        if reduce_cfg is not None:
            num_grads = 1
            np_dtype = np.float32
            self.in_memory_grad_buffer = torch.zeros(
                (num_grads, sum(self.grad_sizes.values())),
                dtype=torch.float32,
                device=f"cuda:{self.rank}",
            )
            device = torch.device(f"cuda:{self.rank}")
            self.h_inv = (
                get_trackstar_preconditioner(
                    self.preprocess_cfg.preconditioner_path,
                    power=-0.5 if self.preprocess_cfg.unit_normalize else -1,
                    device=torch.device(device),
                )
                if self.preprocess_cfg is not None
                else {}
            )
        else:
            num_grads = self.num_items
            np_dtype = convert_dtype_to_np(dtype)
            self.in_memory_grad_buffer = None
            self.h_inv: dict[str, torch.Tensor] = {}

        self.grad_buffer = create_index(
            path,
            num_grads=num_grads,
            grad_sizes=self.grad_sizes,
            dtype=np_dtype,
            with_structure=False,
        )

    def reduce(self, indices: list[int], mod_grads: dict[str, torch.Tensor]):
        assert self.reduce_cfg is not None and self.in_memory_grad_buffer is not None

        # Precondition the gradients
        mod_grads = precondition_grad(mod_grads, self.h_inv)

        unit_normalize = (
            self.preprocess_cfg.unit_normalize
            if self.preprocess_cfg is not None
            else False
        )

        all_grads = torch.cat([mod_grads[m] for m in self.grad_sizes.keys()], dim=-1)
        _reduce(all_grads, self.in_memory_grad_buffer, unit_normalize)

    def __call__(self, indices: list[int], mod_grads: dict[str, torch.Tensor]):
        torch.cuda.synchronize()

        if self.reduce_cfg is not None:
            self.reduce(indices, mod_grads)
        else:
            # It turns out that it's very important for efficiency to write the
            # gradients sequentially instead of first concatenating them, then
            # writing to one vector
            offset = 0
            for module_name in self.grad_sizes.keys():
                self.grad_buffer[
                    indices, offset : offset + mod_grads[module_name].shape[1]
                ] = tensor_to_numpy(mod_grads[module_name])
                offset += mod_grads[module_name].shape[1]

    def teardown(self):
        self.flush()

        if self.reduce_cfg is None:
            return

        assert self.in_memory_grad_buffer is not None

        self.in_memory_grad_buffer = self.in_memory_grad_buffer.cuda()

        if dist.is_initialized():
            dist.reduce(self.in_memory_grad_buffer, dst=0, op=dist.ReduceOp.SUM)

        if self.reduce_cfg.method == "mean":
            self.in_memory_grad_buffer /= self.num_items

        # Unit normalize the reduced gradient
        if self.reduce_cfg.normalize_reduced_grad:
            device = self.in_memory_grad_buffer.device
            self.in_memory_grad_buffer = normalize_flat_grad(
                self.in_memory_grad_buffer, device
            )

        self.in_memory_grad_buffer = self.in_memory_grad_buffer.cpu()

        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank == 0:
            self.grad_buffer[:] = tensor_to_numpy(self.in_memory_grad_buffer).astype(
                self.grad_buffer.dtype
            )

        self.in_memory_grad_buffer = self.in_memory_grad_buffer.cpu()


def create_builder(
    data: Dataset,
    grad_sizes: dict[str, int],
    dtype: torch.dtype,
    *,
    attribute_tokens: bool = False,
    path: Path | None = None,
    reduce_cfg: ReduceConfig | None = None,
    preprocess_cfg: PreprocessConfig | None = None,
) -> Builder:
    """Create the appropriate :class:`Builder` subclass.

    Dispatches based on *attribute_tokens* and *path*:

    * ``path`` given + ``attribute_tokens`` → :class:`TokenBuilder`
    * ``path`` given                        → :class:`SequenceBuilder`
    * no ``path`` + ``attribute_tokens``    → :class:`InMemoryTokenBuilder`
    * no ``path``                           → :class:`InMemorySequenceBuilder`
    """
    if path is not None:
        if attribute_tokens:
            return TokenBuilder(data, grad_sizes, dtype, path=path)
        return SequenceBuilder(
            data,
            grad_sizes,
            dtype,
            path=path,
            reduce_cfg=reduce_cfg,
            preprocess_cfg=preprocess_cfg,
        )
    if attribute_tokens:
        return InMemoryTokenBuilder(data, grad_sizes, dtype)
    return InMemorySequenceBuilder(
        data,
        grad_sizes,
        dtype,
        reduce_cfg=reduce_cfg,
        preprocess_cfg=preprocess_cfg,
    )
