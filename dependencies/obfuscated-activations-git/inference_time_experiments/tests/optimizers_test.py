import pytest
import torch

from obf_reps.optimize.optimizers import (
    AdamContinuousOptimizer,
    GCGOptimizer,
    SignedContinuousOptimizer,
)


@pytest.fixture
def params():
    return [torch.rand(10, 10, requires_grad=True)]


# @pytest.mark.skip(reason="Not implemented yet")
def test_adam_continuous_optimizer(params):
    optimizer = AdamContinuousOptimizer(params, lr=0.01)

    optimizer.step()


@pytest.mark.skip(reason="Not implemented yet")
def test_signed_continuous_optimizer(params):
    optimizer = SignedContinuousOptimizer(params)

    optimizer.step()


@pytest.mark.skip(reason="Not implemented yet")
def test_gcg_optimizer(params):
    optimizer = GCGOptimizer(params)

    optimizer.step()
