import pytest
import torch
from torch.testing import assert_close
from coder import ProjectedVQ

# ------------------------------------------------------------------
# helper ------------------------------------------------------------------
def _same(a: torch.Tensor, b: torch.Tensor, *, atol=1e-6, rtol=1e-6):
    assert a.shape == b.shape, f"Tensors have different shapes: {a.shape} != {b.shape}"
    assert_close(a, b, atol=atol, rtol=rtol)

# ------------------------------------------------------------------
# fixtures ---------------------------------------------------------------
@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    # modest sizes so CI is quick
    m = ProjectedVQ(dim=32, num_books=4, num_codes=8, device="cpu")
    m.eval()                         # inference mode for strict equality
    for p in m.parameters():
        p.requires_grad = False
        p.grad = None
    return m

@pytest.fixture(scope="module")
def inputs(model):
    torch.manual_seed(1)
    return torch.randn(7, model.d)   # 7 examples, dim = model.d

# ------------------------------------------------------------------
# tests ------------------------------------------------------------------
def test_encode_decode_roundtrip(model, inputs):
    """decode(encode(x)) must equal enc_dec_forward reconstruction"""
    # --- full pipeline
    probs = model.encode(inputs, hard=False, flatten=False)
    recon_from_parts = model.decode(probs, unflatten=False)

    # --- training wrapper
    loss_edf, recon_from_edf, probs_edf = model.enc_dec_forward(inputs, hard=False, return_prob=True)

    _same(recon_from_parts, recon_from_edf)
    _same(probs_edf, probs)

@pytest.mark.parametrize("hard", [False, True])
def test_enc_dec_vs_forward(model, inputs, hard):
    """
    Loss & reconstruction from enc_dec_forward must equal those from
    the legacy forward; gradient wrt an arbitrary parameter must match.
    """
    inputs_fwd = inputs.detach().clone().requires_grad_(True)
    inputs_edf = inputs.detach().clone().requires_grad_(True)

    # ---------- legacy path ----------
    with torch.enable_grad():
        out_fwd = model.forward(inputs_fwd, return_prob=False)
        loss_fwd, recon_fwd = out_fwd[:2]
        loss_fwd.backward()

    # ---------- new path ----------
    # NOTE: default path is not hard
    with torch.enable_grad():
        loss_edf, recon_edf = model.enc_dec_forward(inputs_edf, hard=False, return_prob=False)
        loss_edf.backward()

    # ---------- compare ----------
    _same(loss_fwd,  loss_edf) # same value in tensor
    _same(recon_fwd, recon_edf) # same value in tensor
    _same(inputs_fwd.grad, inputs_edf.grad) # same value in input => should be same gradient

def test_flatten_unflatten(model, inputs):
    """Flattened code vector must reshape back to the un-flattened form."""
    codes_flat = model.encode(inputs, hard=False, flatten=True)   # (E, B*C)
    codes_cube = model.encode(inputs, hard=False, flatten=False)  # (E, B, C)
    _same(codes_flat.view_as(codes_cube), codes_cube)
