from __future__ import annotations

"""
Almost entirely copied from `quadratic.ipynb` and trying to train a
multi-token linear predictor for SAE error.
"""
import time
import click
import contextlib
from pathlib import Path
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

def get_dataset(
        dataset_name: str = "stas/openwebtext-10k",
        batch_size: Optional[int] = None,
        tokenizer: Optional[AutoTokenizer] = None,
        # Context setup information
        context_length: int = 128,
    ):
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
        max_length=context_length,  # sae.cfg.context_size,
        add_bos_token=True,  # sae.cfg.prepend_bos,
    )["tokens"]
    tokens = tokens.to("cuda")  # eh
    if batch_size is not None:
        tokens = tokens[:batch_size]
    return tokens

def get_activations(
        model_name: str = "google/gemma-2-2b",
        layer: int = 20,
        return_raw: bool = False,
        n_batch: int = 10_000,
        # Context setup information
        context_length: int = 500,
        inference_batch_size: int = 100,
    ):
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
        tokens = get_dataset(
            dataset_name=dataset_name,
            batch_size=n_batch,
            tokenizer=tokenizer,
            # Context setup information
            context_length=context_length,
        )
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
        for i in tqdm.trange(0, tokens.shape[0], inference_batch_size, desc=f"Forward pass @ inference batch size={inference_batch_size}"):
            j = min(i + inference_batch_size, tokens.shape[0])
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
    print("activations shape on return=", activations.shape)  # These are the ones we will use to understand our SAE; fmt: skip
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

def apply_sae(
    activations: Float[Tensor, "n_samples n_features"],
    sae: SAE,
    batch_size: int,
) -> Float[Tensor, "n_samples n_features"]:
    recons = []
    activations_device = activations.device
    sae_device = next(p.device for p in sae.parameters())
    for i in tqdm.trange(
        0, activations.shape[0], batch_size, desc=f"SAE forward (application) @ batch size={batch_size}"
    ):
        j = min(i + batch_size, activations.shape[0])
        activations_batch = activations[i:j]
        activations_batch = activations_batch.to(sae_device)
        # print("activations_batch.shape:", activations_batch.shape) # DEBUG
        # print("sae_parameters:", {k: str(v.shape) for k, v in sae.named_parameters()}) # DEBUG
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
    # print("unfolding tensor.shape:", tensor.shape) # DEBUG
    # Window sizes:  80%|████████  | 4/5 [00:03<00:00,  1.05it/s]
    # Window sizes:  80%|████████  | 4/5 [00:03<00:00,  1.15it/s]
    # Traceback (most recent call last):
    # File "/mnt/align4_drive2/adrianoh/git2/neel-nanda-mats-2025/LLMDensityestimation/quadratic_linear_sliding_window.py", line 569, in main
    #     r2s = get_sliding_windows_r2(
    #         ^^^^^^^^^^^^^^^^^^^^^^^
    # File "/mnt/align4_drive2/adrianoh/git2/neel-nanda-mats-2025/LLMDensityestimation/quadratic_linear_sliding_window.py", line 489, in get_sliding_windows_r2
    #     sliding_activations = create_sliding_windows(activations, window_size)
    #                         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    # File "/mnt/align4_drive2/adrianoh/git2/neel-nanda-mats-2025/LLMDensityestimation/quadratic_linear_sliding_window.py", line 404, in create_sliding_windows
    #     windows = tensor.unfold(size=window_size, step=1, dimension=1)
    #             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    # RuntimeError: maximum size for tensor at dimension 1 is 18 but size is 20
    windows = tensor.unfold(size=window_size, step=1, dimension=1)
    expected_shape = (batch, seq - window_size + 1, d_model, window_size)
    assert windows.shape == expected_shape, f"windows.shape: {windows.shape}, expected_shape: {expected_shape}" # fmt: skip
    windows = einops.rearrange(windows, "b s d w -> (b s) (d w)")
    assert windows.shape == (batch * (seq - window_size + 1), d_model * window_size), f"windows.shape: {windows.shape}" # fmt: skip
    return windows

def get_sliding_windows_r2(
        model_name: str,
        layer: int,
        window_sizes: list[int],
        reg_lambdas: list[float],
        use_recons: bool = True,
        debug: bool = False,
        # Context setup information
        start_context_at: int = 200,
        end_context_at: int = -10,
        context_length: int = 500,
        sae_batch_size: int = 500,
        probe_batch_size: int = 1024,
        inference_batch_size: int = 100,
    ) -> list[float]:
    #### 1. GET ACTIVATIONS ####
    if not debug:
        print("="*50 + " GETTING REAL ACTIVATIONS " + "="*50)
        activations, is_special_activations, _ = get_activations(
            model_name=model_name,
            layer=layer,
            return_raw=True,
            # Context setup information
            context_length=context_length,
            inference_batch_size=inference_batch_size,
        )
        activations_prev_layer, is_special_activations_prev_layer, _ = get_activations(
            model_name=model_name,
            layer=layer-1,
            return_raw=True,
            # Context setup information
            context_length=context_length,
            inference_batch_size=inference_batch_size,
        )
    else:
        print("="*50 + " USING FAKE (DEBUG) ACTIVATIONS " + "="*50)
        # actiations = get_activations(model_name="google/gemma-2-2b", layer=20, return_raw=True, n_batch=10)
        # print(actiations[0].shape) # FADFASDFASFD
        d_model = 2304 if model_name == "google/gemma-2-2b" else 768 if model_name == "gpt2" else None
        assert d_model is not None, f"d_model: {d_model}, model_name: {model_name}"
        # activations = torch.randn(1024, 256, d_model) # DEBUG
        # activations_prev_layer = torch.randn(1024, 256, d_model) # DEBUG
        activations, is_special_activations, _ = get_activations(
            model_name=model_name,
            layer=layer,
            return_raw=True,
            # Context setup information
            context_length=context_length,
            n_batch=500, # <---- this is the debug mode effect (around 20x smaller)
            inference_batch_size=inference_batch_size,
        )
        activations_prev_layer, is_special_activations_prev_layer, _ = get_activations(
            model_name=model_name,
            layer=layer-1,
            return_raw=True,
            # Context setup information
            context_length=context_length,
            n_batch=500, # <---- this is the debug mode effect (around 20x smaller)
            inference_batch_size=inference_batch_size,
        )
    # Exclude EOS/BOS
    activations = activations[:, start_context_at:end_context_at, :]
    activations_prev_layer = activations_prev_layer[:, start_context_at:end_context_at, :]
    
    # Count how many special activations there are so we can take this into account tbh
    print("================")
    print("How many activations are special tokens?")
    print("================")
    is_special_activations = is_special_activations[:, start_context_at:end_context_at, :]
    is_special_activations_prev_layer = is_special_activations_prev_layer[:, start_context_at:end_context_at, :]
    n_activations_special = torch.sum(is_special_activations).item()
    n_activations_prev_layer_special = torch.sum(is_special_activations_prev_layer).item()
    n_activations_tot = activations.shape[0] * activations.shape[1]
    frac_activations_special = n_activations_special / n_activations_tot
    frac_activations_prev_layer_special = n_activations_prev_layer_special / n_activations_tot
    print(f"n_activations_special={n_activations_special}, n_activations_prev_layer_special={n_activations_prev_layer_special}, n_activations_tot={n_activations_tot}")
    print(f"frac_activations_special={frac_activations_special}, frac_activations_prev_layer_special={frac_activations_prev_layer_special}")
    print("================")
    

    #
    batch, seq, d_model = activations.shape

    #### 2. GET RESIDUALS ####
    print("="*50 + " GETTING SAE RECONSTRUCTION " + "="*50)
    sae = get_sae(model_name=model_name, layer=layer).cuda()
    recons = apply_sae(activations.reshape(batch * seq, d_model), sae, batch_size=sae_batch_size)
    recons = recons.reshape(batch, seq, d_model)
    if use_recons:
        recons, activations = activations, recons
    assert recons.shape == activations.shape
    residuals = activations - recons
    del sae
    del recons # TODO(Adriano) sanity check this maybe please?
    gc.collect()
    torch.cuda.empty_cache()


    print("="*50 + " GETTING R2 SCORES " + "="*50)
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
            assert isinstance(reg_lambda, float), f"reg_lambda: {reg_lambda}, type: {type(reg_lambda)}"
            assert isinstance(window_size, int), f"window_size: {window_size}, type: {type(window_size)}"
            assert isinstance(use_recons, bool), f"use_recons: {use_recons}, type: {type(use_recons)}"
            try:
                linear_fmap = LinearFeatureMap(include_bias=True)
                linear_probe = NonLinearProbe(linear_fmap, reg_lambda=reg_lambda, device="cuda")
                linear_probe.fit(X, y, batch_size=probe_batch_size)
                r2_score = linear_probe.r2_batched(X, y, batch_size=probe_batch_size)
                print(f"r2_score @ model={model_name}, layer={layer}, window_size={window_size}, reg_lambda={reg_lambda}, use_recons={use_recons}: {r2_score}")
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
@click.command()
@click.option("--debug", "-d", is_flag=True, help="Use debug mode")
@click.option("--model", "-m", type=str, help="Model name", default="")
@click.option("--layer-start", "-lstart", type=int, help="Layer start", default=1)
@click.option("--layer-end", "-lend", type=int, help="Layer end", default=-1)
def main(debug: bool, model: str, layer_start: int, layer_end: int):
    # TODO(Adriano) support multi-GPU deployment please!
    # (and large streaming datasets of activations)
    output_folder = Path("quadratic_linear_sliding_window_results")
    output_folder.mkdir(parents=True, exist_ok=False)
    output_log_outfile = output_folder / "stdout.log"
    output_log_errfile = output_folder / "stderr.log"
    models = ["google/gemma-2-2b", "gpt2"] if model == "" else [model]

    # Set the layers to go over
    if layer_end == -1:
        layer_end = 26 if model == "google/gemma-2-2b" else 12
    if layer_start == -1:
        layer_start = 1
    model2layersets = {
        "google/gemma-2-2b": list(range(layer_start, layer_end)),
        "gpt2": list(range(layer_start, layer_end)),
    }
    layersets = [model2layersets[m] for m in models]
    with open(output_log_outfile, "w") as f_out:
        with open(output_log_errfile, "w") as f_err:
            with contextlib.redirect_stdout(f_out):
                with contextlib.redirect_stderr(f_err):
                    # NOTE: we skip the first layer since we use a previous layer for prediction too
                    try:
                        del activations
                    except:
                        pass
                    try:
                        del residuals
                    except:
                        pass
                    # TODO(Adriano) parameterize this stuff
                    reg_lambdas = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1]
                    window_sizes = [1, 3, 5, 10, 20]
                    use_recons_sets = [True, False]
                    #
                    # In the paper they say 
                    # We use 300 contexts of 1024 tokens from the uncopywrited subset of the Pile (Gao et al., 2020) and then
                    # filter to only activations of tokens after position 200 in each context, as Lieberum et al. (2024) find that
                    # earlier tokens are easier for sparse autoencoders to reconstruct, and we wish to ignore the effect of token
                    # position on our results. This results in a dataset of about 247k activations. For linear regressions, we use
                    # a random subset of size 150k as training examples (since all models have a dimension of less than 5000,
                    # this prevents overfitting) and report the R2 on the other 97k activations. For linear transformations to a
                    # multi-dimensional output, we report the average R2 across dimensions. We include bias terms in our linear
                    # regressions but omit them from equations for simplicit
                    #
                    # NOTE however for JBLoom SAEs this won't work since they SUCK at later context, so we need to focus
                    # on earlier contexts
                    start_context_at = 200 if model == "google/gemma-2-2b" else 1 if model == "gpt2" else None
                    end_context_at = -10
                    context_length = 300 if model == "google/gemma-2-2b" else 128 if model == "gpt2" else None
                    # Batch sizes are critial to not OOM
                    # TODO(Adriano) why does batch size shrink SO DAMN MUCH?
                    inference_batch_size = 10 if model == "google/gemma-2-2b" else 1024 if model == "gpt2" else None
                    probe_batch_size = 1024
                    sae_batch_size = 500
                    # ...
                    # ...
                    assert start_context_at is not None and end_context_at is not None and context_length is not None and inference_batch_size is not None and probe_batch_size is not None and sae_batch_size is not None, f"start_context_at: {start_context_at}, end_context_at: {end_context_at}, context_length: {context_length}, inference_batch_size: {inference_batch_size}, probe_batch_size: {probe_batch_size}, sae_batch_size: {sae_batch_size}, model: {model}" # fmt: skip
                    for use_recons in use_recons_sets:
                        for model, layers in zip(models, layersets):
                            for layer in layers:
                                try:
                                    print("=" * 50 + " CALCULATING R2S " + "=" * 50)
                                    hit1 = False
                                    while inference_batch_size >= 1 and not hit1:
                                        hit1 = inference_batch_size == 1
                                        try:
                                            r2s = get_sliding_windows_r2(
                                                model_name=model,
                                                layer=layer,
                                                window_sizes=window_sizes,
                                                reg_lambdas=reg_lambdas,
                                                use_recons=use_recons,
                                                debug=debug,
                                                # Context setup information
                                                start_context_at=start_context_at,
                                                end_context_at=end_context_at,
                                                context_length=context_length,
                                                # Batch sizes... (fkin OOM kms)
                                                inference_batch_size=inference_batch_size,
                                                probe_batch_size=probe_batch_size,
                                                sae_batch_size=sae_batch_size,
                                            )
                                            print(r2s)
                                            with open(output_folder / f"r2s_{model.replace('/', '_')}_{layer}_{'recons' if use_recons else 'no_recons'}.json", "w") as f:
                                                json.dump(r2s, f, indent=4)
                                            print("=" * 50 + " DONE WRITING R2S " + "=" * 50)
                                            gc.collect()
                                            torch.cuda.empty_cache()
                                            for _ in tqdm.trange(10, desc="Sleeping for 10 seconds (please clear cache plz plz)"):
                                                time.sleep(1)
                                            break
                                        except:
                                            inference_batch_size //= 2
                                            inference_batch_size = max(1, inference_batch_size)
                                            print(f"OOM, trying again with batch size={inference_batch_size}")
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
