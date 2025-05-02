from __future__ import annotations

"""
Almost entirely copied from `quadratic.ipynb` and trying to train a
multi-token linear predictor for SAE error.
"""

import gc
from transformers import AutoModelForCausalLM, AutoTokenizer
from sae_lens import SAE  # pip install sae-lens
import torch
from jaxtyping import Float
from torch import Tensor
from typing import List, Tuple, Optional
import tqdm
from transformer_lens.utils import tokenize_and_concatenate
from datasets import load_dataset
import torch
from torch import Tensor
from typing import Callable, Optional
import einops
import json
import itertools
import traceback

def get_sae(model_name: str = "google/gemma-2-2b", layer: int = 20):
    assert model_name in ["gpt2", "google/gemma-2-2b"]
    with torch.no_grad():
        # 1. Fetch the SAE
        assert model_name in ["gpt2", "google/gemma-2-2b"]
        if model_name == "gpt2":
            sae, cfg_dict, sparsity = SAE.from_pretrained(
                    release = "gpt2-small-res-jb",
                    sae_id = f"blocks.{layer}.hook_resid_pre",
                    device = "cuda"
                )
        else:
            sae, cfg_dict, sparsity = SAE.from_pretrained(
                release="gemma-scope-2b-pt-res-canonical",
                sae_id=f"layer_{layer}/width_16k/canonical",
            )
        sae.cuda()
        sae.eval()
        for p in sae.parameters():
            p.requires_grad = False
            p.grad = None
        return sae
def get_model(model_name: str = "google/gemma-2-2b"):
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cuda")
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
        p.grad = None
    return model
    
def get_tokenizer(model_name: str = "google/gemma-2-2b"):
    return AutoTokenizer.from_pretrained(model_name)

def get_dataset(dataset_name: str = "stas/openwebtext-10k", batch_size: Optional[int] = None, tokenizer: Optional[AutoTokenizer] = None):
    if tokenizer is None:
        raise ValueError("Tokenizer is required for `get_dataset`")
    dataset_name = "stas/openwebtext-10k"  # yolo
    dataset = load_dataset(
        dataset_name, split="train", trust_remote_code=True
    )  # Smaller version
    tokens = tokenize_and_concatenate(
        dataset=dataset,  # type: ignore
        tokenizer=tokenizer,  # type: ignore
        streaming=True,
        # NOTE: all these have context 128
        max_length=128,  # sae.cfg.context_size,
        add_bos_token=True,  # sae.cfg.prepend_bos,
    )["tokens"]
    tokens = tokens.to("cuda")  # eh
    if batch_size is not None:
        tokens = tokens[:batch_size]
    return tokens

def get_activations(model_name: str = "google/gemma-2-2b", layer: int = 20, return_raw: bool = False, n_batch: int = 10_000):
    torch.set_grad_enabled(False)
    with torch.no_grad():
        """
        Load the things...
        """
        # 1. Get our model tokenizer etc...
        # sae = get_sae(model_name, layer) # Not used yet lmao
        model = get_model(model_name)
        tokenizer = get_tokenizer(model_name)

        # GEt the full dataset
        dataset_name = "stas/openwebtext-10k"  # yolo
        tokens = get_dataset(dataset_name, batch_size=n_batch, tokenizer=tokenizer)

    """
    Collect some output activations.
    """
    collected_outputs = []
    def gather_target_act_hook(mod, inputs, outputs):
        nonlocal collected_outputs
        assert isinstance(outputs, tuple)
        assert isinstance(outputs[0], torch.Tensor), f"Expected a single tensor output, got {outputs}" # fmt: skip
        collected_outputs.append(outputs[0].detach().requires_grad_(False).cpu())
        return outputs


    if model_name == "gpt2":
        handle = model.transformer.h[layer].register_forward_hook(gather_target_act_hook)
    else:
        handle = model.model.layers[layer].register_forward_hook(gather_target_act_hook)
    try:
        batch_size = 100
        for i in tqdm.trange(0, tokens.shape[0], batch_size):
            j = min(i + batch_size, tokens.shape[0])
            model.forward(tokens[i:j])
    finally:
        handle.remove()
    collected_outputs = torch.cat(collected_outputs, dim=0)
    print(collected_outputs.shape)

    tokens_is_special = (
        (tokens == tokenizer.bos_token_id)
        | (tokens == tokenizer.eos_token_id)
        | (tokens == tokenizer.pad_token_id)
    )
    if return_raw:
        assert tokens.shape == collected_outputs.shape[:-1], f"tokens.shape: {tokens.shape}, collected_outputs.shape: {collected_outputs.shape}"
        assert collected_outputs.shape[:-1] == tokens_is_special.shape, f"collected_outputs.shape: {collected_outputs.shape}, tokens_is_special.shape: {tokens_is_special.shape}"
        return collected_outputs, tokens_is_special, tokens
    tokens_is_special_flat = tokens_is_special.cpu().flatten()
    collected_outputs_flat = collected_outputs.cpu().reshape(
        -1, collected_outputs.shape[-1]
    )
    activations = collected_outputs_flat[~tokens_is_special_flat, :]
    print(activations.shape)  # These are the ones we will use to understand our SAE
    return activations

class QuadraticFeatureMap:
    """Second‑order (quadratic) feature expansion.

    Given activations X ∈ ℝ^{N×d}, returns Φ(X) that concatenates an optional bias,
    the original linear terms, and the unique quadratic terms x_i x_j with i ≤ j.
    Optionally subsamples quadratic terms to keep dimensionality manageable.

    By o3.
    """

    def __init__(
        self,
        include_bias: bool = True,
        include_linear: bool = True,
        max_quadratic_features: Optional[int] = None,
    ) -> None:
        self.include_bias = include_bias
        self.include_linear = include_linear
        self.max_quadratic_features = max_quadratic_features
        self._tri_idx_cache = {}

    def _upper_tri_indices(self, d: int, device: torch.device):
        # Cache indices so we don't re‑allocate on every call
        if (d, device) not in self._tri_idx_cache:
            self._tri_idx_cache[(d, device)] = torch.triu_indices(d, d, device=device)
        return self._tri_idx_cache[(d, device)]

    def __call__(self, x: Tensor) -> Tensor:
        """Compute the quadratic feature map.

        Args:
            x: (N, d) activations.
        Returns:
            Φ(x): (N, D) transformed feature matrix.
        """
        if x.dim() != 2:
            raise ValueError("Input must have shape (N, d)")
        N, d = x.shape
        parts = []
        if self.include_bias:
            parts.append(torch.ones(N, 1, device=x.device, dtype=x.dtype))
        if self.include_linear:
            parts.append(x)

        # Quadratic terms – only keep i ≤ j to avoid duplicates
        tri_i, tri_j = self._upper_tri_indices(d, x.device)
        quad_terms = x.unsqueeze(2) * x.unsqueeze(1)  # (N, d, d)
        quad_terms = quad_terms[:, tri_i, tri_j]      # (N, d(d+1)/2)

        if self.max_quadratic_features is not None and quad_terms.shape[1] > self.max_quadratic_features:
            # Uniform random subsample to cap dimensionality
            idx = torch.randperm(quad_terms.shape[1], device=x.device)[: self.max_quadratic_features]
            quad_terms = quad_terms[:, idx]
        parts.append(quad_terms)
        return torch.cat(parts, dim=1)


class NonLinearProbe(torch.nn.Module):
    """
    Generic non‑linear probe: Φ(·) → linear ridge regression.

    By o3
    """

    def __init__(
        self,
        feature_map: Callable[[Tensor], Tensor],
        reg_lambda: float = 1e-4,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__()
        self.feature_map = feature_map
        self.reg_lambda = reg_lambda
        self.device = torch.device(device)
        self.weight: Optional[Tensor] = None  # (D, k)

    def fit(self, X: Tensor, y: Tensor, batch_size: Optional[int] = None) -> None:
        """Fit ridge regression weights.

        Args:
            X: (N, d) activations.
            y: (N, k) targets (e.g., SAE residuals).
        """
        if batch_size is not None:
            return self.fit_batched(X, y, batch_size)
        X, y = X.to(self.device), y.to(self.device)
        Φ = self.feature_map(X)  # (N, D)
        # Closed‑form ridge solution: W = (ΦᵀΦ + λI)^{-1} Φᵀ y
        XtX = Φ.T @ Φ
        if self.reg_lambda > 0:
            XtX += self.reg_lambda * torch.eye(XtX.size(0), device=self.device, dtype=Φ.dtype)
        self.weight = torch.linalg.solve(XtX, Φ.T @ y)
    
    def fit_batched(self, X: Tensor, y: Tensor, batch_size: int = 1024) -> None:
        """Fit ridge regression weights. By Claude.

        Args:
            X: (N, d) activations.
            y: (N, k) targets (e.g., SAE residuals).
            batch_size: int, batch size for processing large datasets.
        """
        X, y = X.to(self.device), y.to(self.device)
        N = X.shape[0]
        assert X.ndim == 2, f"X.shape: {X.shape}"
        
        # Process in batches to avoid memory issues
        XtX = None
        XtY = None
        
        for i in tqdm.trange(0, N, batch_size, desc=f"Fitting probe over batches (batch_size={batch_size})"):
            end_idx = min(i + batch_size, N)
            X_batch = X[i:end_idx]
            y_batch = y[i:end_idx]
            
            # Transform batch
            Φ_batch = self.feature_map(X_batch)  # (batch_size, D)
            assert Φ_batch.ndim == 2, f"Φ_batch.shape: {Φ_batch.shape}"
            assert Φ_batch.shape[0] == y_batch.shape[0], f"Φ_batch.shape: {Φ_batch.shape}, y_batch.shape: {y_batch.shape}"
            
            # print("Φ_batch shape:", Φ_batch.shape) # DEBUG
            # print("y_batch shape:", y_batch.shape) # DEBUG
            # print("X_batch shape:", X_batch.shape) # DEBUG

            # Accumulate statistics
            batch_XtX = Φ_batch.T @ Φ_batch
            batch_XtY = Φ_batch.T @ y_batch
            
            if XtX is None:
                XtX = batch_XtX
                XtY = batch_XtY
            else:
                XtX += batch_XtX
                XtY += batch_XtY
        
        # Closed‑form ridge solution: W = (ΦᵀΦ + λI)^{-1} Φᵀ y
        if self.reg_lambda > 0:
            XtX += self.reg_lambda * torch.eye(XtX.size(0), device=self.device, dtype=XtX.dtype)
        
        self.weight = torch.linalg.solve(XtX, XtY)

    @torch.no_grad()
    def predict(self, X: Tensor) -> Tensor:
        if self.weight is None:
            raise RuntimeError("Probe has not been fitted yet.")
        Φ = self.feature_map(X.to(self.device))
        return Φ @ self.weight  # (N, k)

    # TODO(Adriano) not entirely sure this will be numerically stable ngl
    @torch.no_grad()
    def r2(self, X: Tensor, y: Tensor, batch_size: Optional[int] = None) -> float:
        if batch_size is not None:
            return self.r2_batched(X, y, batch_size)
        y_pred = self.predict(X)
        ss_res = torch.sum((y.to(self.device) - y_pred) ** 2)
        ss_tot = torch.sum((y.to(self.device) - y.mean(dim=0, keepdim=True).to(self.device)) ** 2)
        return 1.0 - ss_res.item() / ss_tot.item()
    
    @torch.no_grad()
    def r2_batched(self, X: Tensor, y: Tensor, batch_size: int = 1024) -> float:
        X, y = X.to(self.device), y.to(self.device)
        N = X.shape[0]
        
        # Calculate y_mean for ss_tot
        y_mean = y.mean(dim=0, keepdim=True)
        
        ss_res = 0.0
        ss_tot = 0.0
        
        for i in tqdm.trange(0, N, batch_size, desc=f"Computing R² over batches (batch_size={batch_size})"):
            end_idx = min(i + batch_size, N)
            X_batch = X[i:end_idx]
            y_batch = y[i:end_idx]
            
            # Get predictions for this batch
            y_pred_batch = self.predict(X_batch)
            
            # Accumulate sum of squared residuals
            ss_res += torch.sum((y_batch - y_pred_batch) ** 2).item()
            
            # Accumulate total sum of squares
            ss_tot += torch.sum((y_batch - y_mean) ** 2).item()
        
        return 1.0 - ss_res / ss_tot

# TESTING
if __name__ == "__main__":
    # Example usage - DEBUG by o3
    torch.manual_seed(0)
    N, d, k = 1024, 64, 32
    X = torch.randn(N, d)
    true_W_quadratic = torch.randn(d + d * (d + 1) // 2 + 1, k) * 0.1  # bias + linear + quad
    true_W_linear = torch.randn(d + 1, k) * 0.1  # linear + bias
    # NOTE the first term of linear will be bias...
    true_W_linear_as_quadratic = torch.cat([true_W_linear, torch.zeros(d*(d+1)//2, k)], dim=0)
    fmap = QuadraticFeatureMap(include_bias=True, include_linear=True)
    Φ = fmap(X)
    print("phi shape:", Φ.shape, f"from bias={1} plus quadratic={d*(d+1)//2} + linear={d}")
    y = Φ @ true_W_quadratic + 0.01 * torch.randn(N, k)  # synthetic targets
    y_linear = Φ @ true_W_linear_as_quadratic + 0.01 * torch.randn(N, k)  # synthetic targets

    probe = NonLinearProbe(fmap, reg_lambda=1e-3)
    probe_batched = NonLinearProbe(fmap, reg_lambda=1e-3)
    probe_on_linear = NonLinearProbe(fmap, reg_lambda=1e-3)
    probe.fit(X, y)
    probe_batched.fit_batched(X, y)
    probe_on_linear.fit(X, y_linear)
    print("R²:", probe.r2(X, y))
    print("R² batched:", probe_batched.r2(X, y))
    print("R² on linear targets:", probe_on_linear.r2(X, y_linear)) # NOTE it should fit this too!
    print("R² batched on linear targets:", probe_on_linear.r2_batched(X, y_linear))

# Create a linear feature map for comparison
class LinearFeatureMap:
    """Linear feature expansion with optional bias.
    
    Given activations X ∈ ℝ^{N×d}, returns X with an optional bias term.

    By Claude.
    """
    
    def __init__(self, include_bias: bool = True) -> None:
        self.include_bias = include_bias
    
    def __call__(self, x: Tensor) -> Tensor:
        """Compute the linear feature map.
        
        Args:
            x: (N, d) activations.
        Returns:
            Φ(x): (N, d+1) or (N, d) transformed feature matrix.
        """
        if x.dim() != 2:
            raise ValueError("Input must have shape (N, d)")
        N, d = x.shape
        
        if self.include_bias:
            bias = torch.ones(N, 1, device=x.device, dtype=x.dtype)
            return torch.cat([bias, x], dim=1)
        else:
            return x

# TESTING
if __name__ == "__main__":
    N, d, k = 1024, 64, 32
    X = torch.randn(N, d)
    true_W_linear = torch.randn(d + 1, k) * 0.1  # linear + bias
    linear_fmap = LinearFeatureMap()
    Φ_linear = linear_fmap(X)
    y_linear = Φ_linear @ true_W_linear + 0.01 * torch.randn(N, k)  # synthetic targets
    y_nonlinear = Φ @ true_W_quadratic + 0.01 * torch.randn(N, k)  # synthetic targets

    # Technically a LINEAR probe - show that it does not work on the quadratic case...
    linear_probe = NonLinearProbe(linear_fmap, reg_lambda=1e-3)
    linear_probe.fit(X, y)
    linear_probe_on_quad = NonLinearProbe(linear_fmap, reg_lambda=1e-3)
    linear_probe_on_quad.fit(X, y_nonlinear)
    linear_probe_on_quad.r2(X, y_nonlinear)
    print("Linear R²:", linear_probe.r2(X, y))
    print("Linear R² on quadratic targets:", linear_probe_on_quad.r2(X, y_nonlinear))

def apply_sae(
    activations: Float[Tensor, "n_samples n_features"],
    sae: SAE,
    batch_size: int,
) -> Float[Tensor, "n_samples n_features"]:
    recons = []
    activations_device = activations.device
    sae_device = next(p.device for p in sae.parameters())
    for i in tqdm.trange(
        0, activations.shape[0], batch_size, desc="SAE forward (application)"
    ):
        j = min(i + batch_size, activations.shape[0])
        activations_batch = activations[i:j]
        activations_batch = activations_batch.to(sae_device)
        recons.append(sae(activations_batch).to(activations_device))
    for r in recons:
        r.cpu() # TODO(Adriano) fking OOM
    recons_pt = torch.cat(recons, dim=0)
    assert recons_pt.shape == activations.shape
    return recons_pt

def create_sliding_windows(tensor, window_size):
    """
    By Claude, look at https://pytorch.org/docs/stable/generated/torch.nn.Unfold.html
    """
    batch, seq, d_model = tensor.shape
    
    # Create windows along the sequence dimension
    # Channel dimension should not be touched
    # 1D spatial dimension
    # tensor = einops.rearrange(tensor, "b s d -> b d s")
    windows = tensor.unfold(size=window_size, step=1, dimension=1)
    expected_shape = (batch, seq - window_size + 1, d_model, window_size)
    assert windows.shape == expected_shape, f"windows.shape: {windows.shape}, expected_shape: {expected_shape}" # fmt: skip
    windows = einops.rearrange(windows, "b s d w -> (b s) (d w)")
    assert windows.shape == (batch * (seq - window_size + 1), d_model * window_size), f"windows.shape: {windows.shape}" # fmt: skip
    return windows

def get_sliding_windows_r2(model_name: str, layer: int, window_sizes: list[int], reg_lambdas: list[float], use_recons: bool = True) -> list[float]:
    #### 1. GET ACTIVATIONS ####
    activations, _, _ = get_activations(model_name=model_name, layer=20, return_raw=True)#, n_batch=100)
    activations_prev_layer, _, _ = get_activations(model_name=model_name, layer=21, return_raw=True)#, n_batch=100)
    # Exclude EOS/BOS
    activations = activations[:, 1:-10, :]
    activations_prev_layer = activations_prev_layer[:, 1:-10, :]
    #
    batch, seq, d_model = activations.shape

    #### 2. GET RESIDUALS ####
    sae = get_sae(model_name=model_name, layer=layer).cuda()
    recons = apply_sae(activations.reshape(batch * seq, d_model), sae, batch_size=1024)
    recons = recons.reshape(batch, seq, d_model)
    if use_recons:
        recons, activations = activations, recons
    assert recons.shape == activations.shape
    residuals = activations - recons
    del sae
    del recons # TODO(Adriano) sanity check this maybe please?
    gc.collect()
    torch.cuda.empty_cache()


    r2_scores = []
    for window_size in tqdm.tqdm(window_sizes, desc="Window sizes"):
        #### 3. CREATE SLIDING WINDOWS ####
        sliding_activations = create_sliding_windows(activations, window_size)
        residuals_last = create_sliding_windows(residuals, window_size)
        activations_prev_layer_last = create_sliding_windows(activations_prev_layer, window_size)
        residuals_last = residuals_last.reshape(-1, d_model, window_size)[:, :, -1]
        activations_prev_layer_last = activations_prev_layer_last.reshape(-1, d_model, window_size)[:, :, -1]

        X = torch.cat([sliding_activations, activations_prev_layer_last], dim=1)
        y = residuals_last

        del sliding_activations
        del residuals_last
        del activations_prev_layer_last
        gc.collect()
        torch.cuda.empty_cache()
        # Don't do the print cuz it's logspam :/
        for reg_lambda in reg_lambdas:#tqdm.tqdm(reg_lambdas, desc="Regularization lambdas"):
            assert isinstance(reg_lambda, float)
            assert isinstance(window_size, int)
            assert isinstance(use_recons, bool)
            try:
                linear_probe = NonLinearProbe(linear_fmap, reg_lambda=reg_lambda, device="cuda")
                linear_probe.fit(X, y, batch_size=1024)
                r2_score = linear_probe.r2_batched(X, y, batch_size=1024)
                # Make sure it's JSON serializeable
                assert isinstance(r2_score, float)
                # Save
                r2_scores.append({"reg_lambda": reg_lambda, "window_size": window_size, "r2_score": r2_score, 'using_recons': use_recons})
            except:
                # Save non entries to denote errors
                r2_scores.append({"reg_lambda": reg_lambda, "window_size": window_size, "r2_score": None, 'using_recons': use_recons})
                print("="*100)
                print(r2_scores[-1])
                print("="*100)
                traceback.print_exc()
                print("="*100)
    return r2_scores

#### TODO(Adriano) train on the REAL activations.... ####
# We use 300 contexts of 1024 tokens from the uncopywrited subset of the Pile (Gao et al., 2020) and then
# filter to only activations of tokens after position 200 in each context, as Lieberum et al. (2024) find that
# earlier tokens are easier for sparse autoencoders to reconstruct, and we wish to ignore the effect of token
# position on our results. This results in a dataset of about 247k activations. 
#
####################################################
def main():
    models = ["google/gemma-2-2b", "gpt2"]
    layersets = [list(range(26)), list(range(12))]
    try:
        del activations
    except:
        pass
    try:
        del residuals
    except:
        pass
    reg_lambdas = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1]
    window_sizes = [1, 3, 5, 10, 20]
    use_recons = [True, False]
    for model, layers in zip(models, layersets):
        for layer in layers:
            try:
                r2s = get_sliding_windows_r2(model, layer, window_sizes, reg_lambdas, use_recons)
                print(r2s)
                with open(f"r2s_{model.replace('/', '_')}_{layer}.json", "w") as f:
                    json.dump(r2s, f, indent=4)
            except:
                print("="*100)
                print(f"error @ model={model}, layer={layer}")
                print("="*100)
                traceback.print_exc()
                print("="*100)
            finally:
                gc.collect()
                torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
