import pytest
import torch
import torch.nn as nn

from bergson import fit_normalizers
from bergson.config import IndexConfig


def test_fit_normalizers_runs(tmp_path, model, dataset):
    target_modules = {
        name
        for name, module in model.base_model.named_modules()
        if isinstance(module, nn.Linear)
    }
    print("len dataset", len(dataset))
    print("target_modules", target_modules)

    dataset = dataset.repeat(10)

    normalizers = fit_normalizers(
        model,
        dataset,
        cfg=IndexConfig(
            run_path=str(tmp_path),
            skip_preconditioners=True,
            normalizer="adam",
        ),
        batches=[[idx] for idx in range(len(dataset))],
        target_modules=target_modules,
    )

    assert len(normalizers) == len(target_modules)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("normalizer_type", ["adam", "adafactor"])
def test_fit_normalizers_with_bias(tmp_path, model, dataset, normalizer_type):
    """fit_normalizers populates bias_avg_sq when include_bias=True."""
    # tiny-Phi3 has no bias on Linear layers; add zero bias
    for m in model.base_model.modules():
        if isinstance(m, nn.Linear) and m.bias is None:
            m.bias = nn.Parameter(torch.zeros(m.out_features, device=m.weight.device))

    target_modules = {
        name
        for name, module in model.base_model.named_modules()
        if isinstance(module, nn.Linear)
    }
    dataset = dataset.repeat(10)

    normalizers = fit_normalizers(
        model,
        dataset,
        cfg=IndexConfig(
            run_path=str(tmp_path / normalizer_type),
            skip_preconditioners=True,
            normalizer=normalizer_type,
            include_bias=True,
        ),
        batches=[[idx] for idx in range(len(dataset))],
        target_modules=target_modules,
    )
    for name, norm in normalizers.items():
        assert norm.bias_avg_sq is not None, f"{name} missing bias_avg_sq"
        assert norm.bias_avg_sq.shape[0] > 0
        assert (norm.bias_avg_sq >= 0).all()  # second moments are non-negative
