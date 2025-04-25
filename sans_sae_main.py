from __future__ import annotations
import click
import gc
import math
import re
from typing import List, Optional
import tqdm
import numpy as np
import torch
import os
from pathlib import Path
import json
from datasets import load_dataset
import dotenv
from transformers import AutoTokenizer
from transformer_lens.utils import tokenize_and_concatenate
import tqdm
import matplotlib.pyplot as plt
import shutil
import torch.nn.functional as F
from jaxtyping import Float, Int
from transformer_lens import ActivationCache
from transformer_lens.components import TransformerBlock, LayerNormPre
from sae_lens import HookedSAETransformer, SAE

# load stuff from our own code
from sans_sae_lib.schemas import ExtractedActivations, FlattenedExtractedActivations
from sans_sae_lib.utils import plot_cosine_kernel, plot_all_nc2_top_pcs, plot_all_nc2_top_pcs_errs


class ResidAndLn2Comparer:
    """
    A class that basically automates the task of taking in a dataset, tokenizing it, running it through
    GPT2 using HookedSAETransformer with JBloom's SAEs.

    It is meant for measuring/plotting the impacts on the activations on the model.
    The outputs of this module are/should be:
    1. Plots of histograms for PCA applied on
        - The residual stream activations
        - The ln2 activations
        - The SAE processed residual stream activations
        - The SAE processed ln2 activations
        (there are 2D histograms for any pair of PCs in the top 10 PCs)
    2. Plots of the MSE, FVU, etc... for the SAEs at each location (including also at ln2) as 1D histograms
    3. Plots of the cosine similarity between the PCs pre and post-sae intervention (a 2D heatmap)
        (this also will include a )
    4. Plots of the cosine similarity between input datapoints and output datapoints pre and post-SAE
        intervention in the form of a 1D histogram.
    5. SAE error norm/mse/fvu (or in the future any general function)
    6. Plots for each K out of a set of top ks of the distribution of reconstructed activations
        (this is critical). To do this with non-top-k-trained SAEs we just force the latents to be top-K'ed
        but ideally we should also support some top-k SAEs.
    
    TODO(Adriano) please add support for:
    1. Labeling points
    2. Removing BOS/EOS/etc... (or they should also be labeled, etc...)
    3. Support plotting other saes/models
    4. Support NOT swapping the SAE dimensionalities post-SAE because that might change
        the perspective and thus make the visual comparison not apples-to-apples.
    """
    def __init__(self):
        # Load the model and set some basic settins
        self.model_name = "gpt2"
        self.sae_release = "gpt2-small-res-jb"
        self.device = "cuda" # NOTE: you should use CUDA_VISIBLE_DEVICES to select the GPU
        self.model = HookedSAETransformer.from_pretrained(self.model_name, device=self.device)
        self.model.eval()

        # Load the SAEs and set helper variables
        self.d_model = self.model.cfg.d_model
        self.n_layers = self.model.cfg.n_layers
        self.load_jbloom_gpt2_saes(sae_release=self.sae_release)
        self.forced_ks: List[int] = [1, 2, 4, 8, 16, 32, 64, 128, 99999999]

    def load_jbloom_gpt2_saes(self, sae_release: str = "gpt2-small-res-jb"):
        self.saes = []
        self.sae_cfg_dicts = []
        self.sae_sparsities = []
        for layer in range(self.model.cfg.n_layers):
            sae, cfg_dict, sparsity = SAE.from_pretrained(
                release = sae_release,
                sae_id = f"blocks.{layer}.hook_resid_pre",
                device = self.device
            )
            sae.eval()
            self.saes.append(sae)
            self.sae_cfg_dicts.append(cfg_dict)
            self.sae_sparsities.append(sparsity)
    
    def apply_sae(
            self,
            activations: Float[torch.Tensor, "layer batch seq d_model"] | List[Float[torch.Tensor, "batch seq d_model"]],
            force_topk: Optional[int] = None
        ) -> Float[torch.Tensor, "batch seq d_model"]:
        """
        Apply the SAE to the activations.
        """
        # Force topk to be able to scan the effects of narrowing the window of sparsity
        if force_topk is not None:
            layers, batch, seq, d_model = activations.shape
            encoded = torch.stack([self.saes[i].encode(activations[i]) for i in range(len(self.saes))])
            assert encoded.min().item() >= 0, f"encoded.min()={encoded.min().item()}"
            assert encoded.shape[:-1] == activations.shape[:-1], f"encoded.shape[:-1]={encoded.shape[:-1]}, activations.shape[:-1]={activations.shape[:-1]}"
            d_sae = encoded.shape[-1]
            force_topk = min(force_topk, d_sae) # Cannot go above
            topk_values, topk_indices = torch.topk(encoded, k=force_topk, dim=-1)
            assert topk_indices.shape == (layers, batch, seq, force_topk), f"topk_indices.shape={topk_indices.shape}, need [3] = {force_topk}"
            assert topk_values.shape == (layers, batch, seq, force_topk), f"topk_values.shape={topk_values.shape}, need [3] = {force_topk}"
            # By Claude: zero out everything except the topk values
            mask = torch.zeros_like(encoded)
            assert mask.shape == encoded.shape, f"mask.shape={mask.shape}, encoded.shape={encoded.shape}"
            mask.scatter_(-1, topk_indices, topk_values)
            return torch.stack([self.saes[i].decode(mask[i]) for i in range(len(self.saes))])
        # No force: just apply the SAE normally
        return torch.stack([self.saes[i](activations[i]) for i in range(len(self.saes))])

    def get_post_ln2_hookpoint_after_hookpoint(self, hookpoint: str) -> torch.Tensor:
        """
        Look here to see where mlp_in happens:
        ------------------------------------------------------------
        pre:
        https://github.com/TransformerLensOrg/TransformerLens/blob/e65fafb4791c66076bc54ec9731920de1e8c676f/transformer_lens/components/transformer_block.py#L191

        also
        ------------------------------------------------------------
        post:
        https://github.com/TransformerLensOrg/TransformerLens/blob/e65fafb4791c66076bc54ec9731920de1e8c676f/transformer_lens/components/layer_norm_pre.py#L52
        """
        assert re.match(r"^blocks.[0-9]+.hook_resid_pre$", hookpoint)
        layer = int(re.match(r"^blocks.([0-9]+)\.hook_resid_pre$", hookpoint).group(1))
        return f"blocks.{layer}.ln2.hook_normalized"

    def get_hookpoint_layer(self, hookpoint: str) -> int:
        assert re.match(r"^blocks.[0-9]+\..*$", hookpoint)
        return int(re.match(r"^blocks.([0-9]+)\..*$", hookpoint).group(1))

    def get_post_ln2_value_after_hookpoint_from_cache(
            self,
            cache: ActivationCache,
            hookpoint: str
        ) -> Float[torch.Tensor, "layer batch seq d_model"]:
        """
        Quick helper meethod to get the post-ln2 value after a hookpoint in the model (vanilla)
        This is used for the express purpose of being able to calculate the SAE's MSE/FVU impact on a
        cerain later location (right after ln2 right before MLP).
        """
        # 1. Get the ln2 that we will have to apply
        layer_num: int = self.get_hookpoint_layer(hookpoint)
        block: TransformerBlock = self.model.blocks[layer_num]
        assert isinstance(block, TransformerBlock), f"block is {type(block)}"
        ln2 = block.ln2
        assert isinstance(ln2, LayerNormPre), f"ln2 is {type(ln2)}"
        # 2. Get the activations
        post_ln2_hp: str = self.get_post_ln2_hookpoint_after_hookpoint(hookpoint)
        assert post_ln2_hp in cache.keys(), f"post_ln2_hp={post_ln2_hp} not in cache.keys()={cache.keys()}"
        post_ln2_act: torch.Tensor = cache[post_ln2_hp]
        return post_ln2_act
    
    def get_post_ln2_value_after_hookpoint_from_activations(
            self,
            activations: Float[torch.Tensor, "batch seq d_model"],
            hookpoint: str
    ) -> Float[torch.Tensor, "layer batch seq d_model"]:
        """
        Like `get_post_ln2_value_after_hookpoint` but this will instead take in activations
        instead of a cache object. The point is that we will propagate forwards the SAE-processed
        activations instead of the vanilla activations.
        """
        # 1. Get the ln2 that we will have to apply
        layer_num: int = self.get_hookpoint_layer(hookpoint)
        block: TransformerBlock = self.model.blocks[layer_num]
        assert isinstance(block, TransformerBlock), f"block is {type(block)}"
        ln2 = block.ln2
        assert isinstance(ln2, LayerNormPre), f"ln2 is {type(ln2)}"
        # 2. run the model for exactly one layer and hook out the spot right before ln2
        debug_number = torch.randn(1, device=self.device) * 999999
        buffer_post_ln2 = torch.ones_like(activations) * debug_number
        def write_hook(normalized_resid, hook):
            buffer_post_ln2[:] = normalized_resid
            return normalized_resid
        self.model.run_with_hooks(
            activations,
            fwd_hooks=[
                (
                    f"blocks.{layer_num}.ln2.hook_normalized",
                    write_hook,
                ),
            ],
            start_at_layer=layer_num,
            stop_at_layer=layer_num+1,
        )
        # Sanity check that we actually wrote out
        assert not torch.any(buffer_post_ln2 == debug_number), f"buffer_post_ln2 is still {debug_number}"
        return buffer_post_ln2
        
        
    def extract_activations(self, tokens: Int[torch.Tensor, "batch seq"], batch_size: int = 30) -> ExtractedActivations:
        """
        Extracts the activations of the model and the SAEs and returns them as tensors.
        """
        # 1. Define output buffers
        dataset_length, sequence_length = tokens.shape
        debug_numbers = torch.randn(4, device="cpu") * 999999
        sae_ins = torch.ones((len(self.saes), dataset_length, sequence_length, self.d_model), device="cpu") * debug_numbers[0] # fmt: skip
        sae_outs_per_k = torch.ones((len(self.forced_ks), len(self.saes), dataset_length, sequence_length, self.d_model), device="cpu") * debug_numbers[1] # fmt: skip
        ln2s = torch.ones((len(self.saes), dataset_length, sequence_length, self.d_model), device="cpu") * debug_numbers[2] # fmt: skip
        ln2s_saed_per_k = torch.ones((len(self.forced_ks), len(self.saes), dataset_length, sequence_length, self.d_model), device="cpu") * debug_numbers[3] # fmt: skip

        while True:
            try:
                with torch.no_grad():
                    for i in tqdm.trange(0, len(tokens), batch_size, desc=f"Batch Size = {batch_size}"):
                        j = min(i + batch_size, len(tokens))
                        assert j > i, f"j={j} must be greater than i={i}"
                        # activation store can give us tokens.
                        # 1. Get the activations for our current batch of tokens
                        # TODO(Adriano) if you do this with run_with_hooks (etc...) you could
                        # get better performance for sure.
                        batch_tokens = tokens[i:j]
                        _, cache = self.model.run_with_cache(batch_tokens, prepend_bos=True)

                        # 2. Extract the desired activations from the cache
                        # print(cache.keys())
                        # Use the SAE
                        # print(len(extractor.block2sae))
                        # print(f"hook_name={extractor.block2sae[8].cfg.hook_name}") # Nope
                        sae_in = torch.stack([cache[self.saes[x].cfg.hook_name].detach() for x in range(len(self.saes))])
                        _ln2 = torch.stack([self.get_post_ln2_value_after_hookpoint_from_cache(cache, self.saes[x].cfg.hook_name).detach() for x in range(len(self.saes))]) # fmt: skip
                        del cache
                        gc.collect()
                        torch.cuda.empty_cache()
                        assert sae_in.shape == _ln2.shape, f"sae_in.shape={sae_in.shape}, ln2_.shape={_ln2.shape}"
                        # feature_acts = [extractor.block2sae[i].encode(sae_in[i])
                        # TODO(Adriano) this should be possible to parallelize, make faster, etc...
                        _sae_outs_per_k = torch.stack([self.apply_sae(sae_in, force_topk=k) for k in self.forced_ks])
                        assert sae_in.shape == _sae_outs_per_k[0].shape, f"sae_in.shape={sae_in.shape}, sae_outs_per_k[0].shape={_sae_outs_per_k[0].shape}"
                        _ln2_saed_per_k = torch.stack(
                            [
                                torch.stack(
                                    [
                                        self.get_post_ln2_value_after_hookpoint_from_activations(_sae_outs_per_k[y, x], self.saes[x].cfg.hook_name) for x in range(len(self.saes))
                                    ]
                                )
                                # NOTE k index not k
                                for y in range(len(self.forced_ks))
                            ]
                        )

                        # 2. Sanity check the sizes
                        assert sae_in.shape == _sae_outs_per_k[0].shape
                        assert sae_in.shape == _ln2.shape # NOTE: this will not scale to different layers but eh
                        assert _sae_outs_per_k.shape == _ln2_saed_per_k.shape
                        assert sae_in.shape[0] == len(self.saes)
                        assert sae_in.shape[1] == j - i
                        assert sae_in.shape[2] == sequence_length
                        assert sae_in.shape[3] == self.d_model, f"sae_in.shape={sae_in.shape}, need [2] = {self.d_model}" # fmt: skip
                        assert sae_in.ndim == 4

                        # At least one of these must differ
                        assert not all(torch.all(_ln2_saed_per_k[kidx] == _ln2).item() for kidx in range(len(self.forced_ks)))

                        # 3. Store the activations to the appropriate buffers
                        sae_ins[:, i:j, :, :] = sae_in.cpu()
                        sae_outs_per_k[:, :, i:j, :, :] = _sae_outs_per_k.cpu()
                        ln2s[:, i:j, :, :] = _ln2.cpu()
                        ln2s_saed_per_k[:, :, i:j, :, :] = _ln2_saed_per_k.cpu()
                
                # Sanity check that we actually wrote out
                assert not torch.any(sae_ins == debug_numbers[0]), f"sae_ins is still {debug_numbers[0]}"
                assert not torch.any(sae_outs_per_k == debug_numbers[1]), f"sae_outs_per_k is still {debug_numbers[1]}"
                assert not torch.any(ln2s == debug_numbers[2]), f"ln2s is still {debug_numbers[2]}"
                assert not torch.any(ln2s_saed_per_k == debug_numbers[3]), f"ln2s_saed_per_k is still {debug_numbers[3]}"
                
                # Done
                return ExtractedActivations(
                    sae_ins=sae_ins,
                    sae_outs_per_k=sae_outs_per_k,
                    ln2s=ln2s,
                    ln2s_saed_per_k=ln2s_saed_per_k
                )
            except torch.OutOfMemoryError as e:
                # print(type(e), e)
                print(e)
                if batch_size == 1:
                    print("BATCH SIZE IS 1, RAISING ERROR")
                    raise e
                # eh... we should do BINARY search instead of backoff in one direction but eh... easier :)
                backoff_factor = 0.8
                batch_size = max(1, int(math.floor(batch_size * backoff_factor)))
                print(f"REDUCING BATCH SIZE FOR NEXT ITERATION TO {batch_size}")
                # ................
                # Deletion stack can help gc achieve its goals :)
                # (pick things in gpu)
                try:
                    del sae_in
                except NameError:
                    pass
                try:
                    del _ln2
                except NameError:
                    pass
                try:
                    del _sae_outs_per_k
                except NameError:
                    pass
                try:
                    del _ln2_saed_per_k
                except NameError:
                    pass
                # ................
                gc.collect()
                torch.cuda.empty_cache()
                continue

def main_helper_plot_error_value_data(
        comparer: ResidAndLn2Comparer,
        extracted_activations_flattened: FlattenedExtractedActivations,
        global_plot_folder_path: Path
) -> None:
    res_sae_err_norms_output_folder = global_plot_folder_path / "res_sae_err_norms" # fmt: skip
    res_sae_variance_explained_output_folder = global_plot_folder_path / "res_sae_variance_explained" # fmt: skip
    res_sae_mse_output_folder = global_plot_folder_path / "res_sae_mse" # fmt: skip

    # ln2 folders
    ln2_sae_err_norms_output_folder = global_plot_folder_path / "ln2_sae_err_norms" # fmt: skip
    ln2_sae_variance_explained_output_folder = global_plot_folder_path / "ln2_sae_variance_explained" # fmt: skip
    ln2_sae_mse_output_folder = global_plot_folder_path / "ln2_sae_mse" # fmt: skip

    for (name, folder, arr_per_k) in tqdm.tqdm([
        # res
        ("res_sae_err_norms", res_sae_err_norms_output_folder, extracted_activations_flattened.res_sae_error_norms), # fmt: skip
        ("res_sae_variance_explained", res_sae_variance_explained_output_folder, extracted_activations_flattened.res_sae_var_explained), # fmt: skip
        ("res_sae_mse", res_sae_mse_output_folder, extracted_activations_flattened.res_sae_mse), # fmt: skip
        # ln2
        ("ln2_sae_err_norms", ln2_sae_err_norms_output_folder, extracted_activations_flattened.ln2_sae_error_norms), # fmt: skip
        ("ln2_sae_variance_explained", ln2_sae_variance_explained_output_folder, extracted_activations_flattened.ln2_sae_var_explained), # fmt: skip
        ("ln2_sae_mse", ln2_sae_mse_output_folder, extracted_activations_flattened.ln2_sae_mse), # fmt: skip
        
    ]):
        if folder.exists() and len(list(folder.glob("*"))) == 0:
            shutil.rmtree(folder)
        folder.mkdir(parents=True, exist_ok=False)
        # Check that the first two dimensions are k and layer
        assert arr_per_k.shape[0] == len(comparer.forced_ks), f"arr_per_k.shape[0]={arr_per_k.shape[0]}, need {len(comparer.forced_ks)}" # fmt: skip
        assert arr_per_k.shape[1] == len(comparer.saes), f"arr_per_k.shape[1]={arr_per_k.shape[1]}, need {len(comparer.saes)}" # fmt: skip
        for layer in tqdm.trange(len(comparer.saes)):
            for kidx, k in tqdm.tqdm(enumerate(comparer.forced_ks), total=len(comparer.forced_ks), desc=f"layer {layer} x ks"): # fmt: skip
                arr = arr_per_k[kidx, layer]
                assert arr.ndim == 1 # just a bunch of errors, etc...
                filepath = folder / f"layer_{layer}_k_{k}.png"
                plt.hist(arr.cpu().log10().numpy(), bins=100)
                plt.title(f"log10({name}) (layer {layer}) @ top=k {k}")
                plt.savefig(filepath)
                plt.close()
                filepath_meta = folder / f"layer_{layer}_k_{k}.json"
                with open(filepath_meta, "w") as f:
                    json.dump({
                        "layer": layer,
                        "name": name,
                        "shape": str(arr.shape),
                        "min": arr.min().item(),
                        "max": arr.max().item(),
                        "mean": arr.mean().item(),
                        "std": arr.std().item(),
                        "median": arr.median().item(),
                        "q1": arr.quantile(0.25).item(),
                        "q3": arr.quantile(0.75).item(),
                        "k": k
                    }, f, indent=4)

def main_helper_plot_pc_cosine_sims(
    comparer: ResidAndLn2Comparer,
    extracted_activations_flattened: FlattenedExtractedActivations,
    global_plot_folder_path: Path
) -> None:
    # ...
    plots_output_folder = global_plot_folder_path / "plots_cosine_sim_pca_post_sae"
    if plots_output_folder.exists() and len(list(plots_output_folder.glob("*"))) == 0:
        shutil.rmtree(plots_output_folder)
    plots_output_folder.mkdir(parents=True, exist_ok=False)
    # Run the plot generation
    print("\nAnalyzing multiple layers...")
    for layer in tqdm.trange(len(comparer.saes)):
        # Acquire all the principle components we want to compare
        eigenvectors_sae_ins = extracted_activations_flattened.sae_ins_pca_eigenvectors[layer] # fmt: skip
        eigenvectors_ln2s = extracted_activations_flattened.ln2s_pca_eigenvectors[layer] # fmt: skip
        for kidx, k in tqdm.tqdm(enumerate(comparer.forced_ks), total=len(comparer.forced_ks), desc=f"layer {layer} x ks"): # fmt: skip
            eigenvectors_sae_outs = extracted_activations_flattened.sae_outs_per_k_pca_eigenvectors[kidx, layer] # fmt: skip
            eigenvectors_ln2s_saed = extracted_activations_flattened.ln2s_saed_per_k_pca_eigenvectors[kidx, layer] # fmt: skip
            assert eigenvectors_sae_outs.shape == eigenvectors_sae_ins.shape, f"eigenvectors_sae_outs.shape={eigenvectors_sae_outs.shape}, eigenvectors_sae_ins.shape={eigenvectors_sae_ins.shape}" # fmt: skip
            assert eigenvectors_ln2s_saed.shape == eigenvectors_ln2s.shape, f"eigenvectors_ln2s_saed.shape={eigenvectors_ln2s_saed.shape}, eigenvectors_ln2s.shape={eigenvectors_ln2s.shape}" # fmt: skip
            # Save them to the plot folder
            plot_cosine_kernel(eigenvectors_sae_ins, eigenvectors_sae_outs, force_positive=True, save_to_file=plots_output_folder / f"layer_{layer}_k_{k}_res.png") # fmt: skip
            plot_cosine_kernel(eigenvectors_ln2s, eigenvectors_ln2s, force_positive=True, save_to_file=plots_output_folder / f"layer_{layer}_k_{k}_ln2.png") # fmt: skip

def main_helper_plot_distribution_pc_projections(
    comparer: ResidAndLn2Comparer,
    extracted_activations_flattened: FlattenedExtractedActivations,
    global_plot_folder_path: Path
) -> None:
    res_sae_in_pca_histograms_folder = global_plot_folder_path / "res_sae_in_pca_histograms"
    res_sae_out_pca_histograms_folder = global_plot_folder_path / "res_sae_out_pca_histograms"
    ln2_pca_histograms_folder = global_plot_folder_path / "ln2_pca_histograms"
    ln2_sae_effect_pca_histograms_folder = global_plot_folder_path / "ln2_sae_effect_pca_histograms"
    n_pcs = 2 # ehh

    for output_folder, (activations, mean, eigenvectors), per_k in tqdm.tqdm([
        (
            res_sae_in_pca_histograms_folder,
            (
                extracted_activations_flattened.sae_ins,
                extracted_activations_flattened.sae_ins_means,
                extracted_activations_flattened.sae_ins_pca_eigenvectors
            ),
            False
        ),
        (
            res_sae_out_pca_histograms_folder, 
            (
                extracted_activations_flattened.sae_outs_per_k,
                extracted_activations_flattened.sae_outs_per_k_means,
                extracted_activations_flattened.sae_outs_per_k_pca_eigenvectors
            ),
            True
        ),
        (
            ln2_pca_histograms_folder,
            (
                extracted_activations_flattened.ln2s,
                extracted_activations_flattened.ln2s_means,
                extracted_activations_flattened.ln2s_pca_eigenvectors
            ),
            False
        ),
        (
            ln2_sae_effect_pca_histograms_folder,
            (
                extracted_activations_flattened.ln2s_saed_per_k,
                extracted_activations_flattened.ln2s_saed_per_k_means,
                extracted_activations_flattened.ln2s_saed_per_k_pca_eigenvectors
            ),
            True
        )
    ]):
        for layer in tqdm.trange(len(comparer.saes)):
            if per_k:
                for kidx, k in tqdm.tqdm(enumerate(comparer.forced_ks), total=len(comparer.forced_ks), desc=f"layer {layer} x ks"): # fmt: skip
                    output_folder_layer = output_folder / f"layer_{layer}_k_{k}"
                    output_folder_layer.mkdir(parents=True, exist_ok=False)
                    plot_all_nc2_top_pcs(n_pcs, activations[kidx, layer], mean[kidx, layer], eigenvectors[kidx, layer], output_folder_layer)
            else:
                output_folder_layer = output_folder / f"layer_{layer}"
                output_folder_layer.mkdir(parents=True, exist_ok=False)
                plot_all_nc2_top_pcs(n_pcs, activations[layer], mean[layer], eigenvectors[layer], output_folder_layer)

def main_helper_plot_errors_pc_projections(
    comparer: ResidAndLn2Comparer,
    extracted_activations_flattened: FlattenedExtractedActivations,
    global_plot_folder_path: Path
) -> None:
    res_sae_in_pca_histograms_folder = global_plot_folder_path / "res_sae_in_err_pca_histograms"
    res_sae_out_pca_histograms_folder = global_plot_folder_path / "res_sae_out_err_pca_histograms"
    ln2_pca_histograms_folder = global_plot_folder_path / "ln2_err_pca_histograms"
    ln2_sae_effect_pca_histograms_folder = global_plot_folder_path / "ln2_sae_effect_err_pca_histograms"
    n_pcs = 2 # ehh, copy from above lmao
    for output_folder, per_k, (activations, mean, eigenvectors, err_norm, err_var_explained, err_mse) in tqdm.tqdm([
        (
            res_sae_in_pca_histograms_folder,
            # NOTE: per_k refers to whether or not the activations, mean, etc... are per-k; if they are not
            # then they need to be repeated for the zip to work fine (look below)
            False,
            (
                # Projection (binning) data
                extracted_activations_flattened.sae_ins,
                extracted_activations_flattened.sae_ins_means,
                extracted_activations_flattened.sae_ins_pca_eigenvectors,
                # Errors (coloring data)
                extracted_activations_flattened.res_sae_error_norms, # NOTE: this is also per k
                extracted_activations_flattened.res_sae_var_explained, # NOTE: this is also per k
                extracted_activations_flattened.res_sae_mse # NOTE: this is also per k
            )
        ),
        (
            res_sae_out_pca_histograms_folder, 
            True,
            (
                # Projection (binning) data
                extracted_activations_flattened.sae_outs_per_k,
                extracted_activations_flattened.sae_outs_per_k_means,
                extracted_activations_flattened.sae_outs_per_k_pca_eigenvectors,
                # Errors (coloring data)
                extracted_activations_flattened.res_sae_error_norms, # NOTE: this is also per k
                extracted_activations_flattened.res_sae_var_explained, # NOTE: this is also per k
                extracted_activations_flattened.res_sae_mse # NOTE: this is also per k
            )
        ),
        (
            ln2_pca_histograms_folder,
            False,
            (
                # Projection (binning) data
                extracted_activations_flattened.ln2s,
                extracted_activations_flattened.ln2s_means,
                extracted_activations_flattened.ln2s_pca_eigenvectors,
                # Errors (coloring data)
                extracted_activations_flattened.ln2_sae_error_norms, # NOTE: this is also per k
                extracted_activations_flattened.ln2_sae_var_explained, # NOTE: this is also per k
                extracted_activations_flattened.ln2_sae_mse # NOTE: this is also per k
            )
        ),
        (
            ln2_sae_effect_pca_histograms_folder,
            True,
            (
                # Projection (binning) data
                extracted_activations_flattened.ln2s_saed_per_k,
                extracted_activations_flattened.ln2s_saed_per_k_means,
                extracted_activations_flattened.ln2s_saed_per_k_pca_eigenvectors,
                # Errors (coloring data)
                extracted_activations_flattened.ln2_sae_error_norms, # NOTE: this is also per k
                extracted_activations_flattened.ln2_sae_var_explained, # NOTE: this is also per k
                extracted_activations_flattened.ln2_sae_mse # NOTE: this is also per k
            )
        )
    ]):
        for layer in tqdm.trange(len(comparer.saes)):
            for kidx, k in tqdm.tqdm(enumerate(comparer.forced_ks), total=len(comparer.forced_ks), desc=f"layer {layer} x ks"): # fmt: skip
                output_folder_layer = output_folder / f"layer_{layer}_{k}"
                output_folder_layer.mkdir(parents=True, exist_ok=False)
                err_type_names = ["error_norm", "variance_explained", "mse"] # fmt: skip
                # NOTE: errors are ALWAYS per k
                err_arrays = [err_norm[kidx, layer], err_var_explained[kidx, layer], err_mse[kidx, layer]] # fmt: skip
                for err_type_name, err_array in zip(err_type_names, err_arrays):
                    for normalize_by_n_in_bin, normalize_by_n_in_bin_name in zip([False, True], ["unnormalized", "normalized"]):
                        sub_output_folder = output_folder_layer / f"{normalize_by_n_in_bin_name}_{err_type_name}_k_{k}"
                        plot_all_nc2_top_pcs_errs(
                            n_pcs,
                            # Projection parameters
                            # NOTE: this is where the "zip" comes into play
                            activations[kidx, layer] if per_k else activations[layer],
                            mean[kidx, layer] if per_k else mean[layer],
                            eigenvectors[kidx, layer] if per_k else eigenvectors[layer],
                            # Error parameters + Plotting n stuff
                            err_array, # NOTE: already layered
                            normalize_by_n_in_bin, # normalize by in bin but not total
                            not normalize_by_n_in_bin, # normalize by total not in bin
                            # Storage parameters
                            sub_output_folder,
                        )

@click.command()
def main() -> None:
    """
    A short frontend CLI to the same general functionality as `sans_sae.ipynb`. The
    generation of the heatmaps, etc... can take a really long time.
    """
    # 1. Get environment and make sure we will work on a
    # correct GPU
    dotenv.load_dotenv()
    assert "CUDA_VISIBLE_DEVICES" in os.environ, "CUDA_VISIBLE_DEVICES is not set"
    assert len(os.environ["CUDA_VISIBLE_DEVICES"].strip()) > 0, "CUDA_VISIBLE_DEVICES is empty"

    # 2. Get the dataset
    print("="*50 + " [Loading Dataset] " + "="*50) # DEBUG
    # TODO(Adriano) support using the origianl openwebtext
    # dataset = load_dataset("openwebtext", split="train", trust_remote_code=True)
    dataset = load_dataset("stas/openwebtext-10k", split="train", trust_remote_code=True) # Smaller version
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    token_dataset = tokenize_and_concatenate(
        dataset=dataset,  # type: ignore
        tokenizer=tokenizer,  # type: ignore
        streaming=True,
        # NOTE: all these have context 128
        max_length=128, #sae.cfg.context_size,
        add_bos_token=True, #sae.cfg.prepend_bos,
    )
    print("="*50 + " [Loading Model] " + "="*50) # DEBUG
    # TODO(Adriano) do this below...
    # for d in extractor.cfg_dics:
    #     print(d["context_size"]) # NOTE: you should picke the smallest of these...


    # Shorten the dataset for testing more quickly
    dataset_size = 5_000 # should be enough for an initial foray
    token_dataset_short = token_dataset[:dataset_size]['tokens']
    dataset_length = token_dataset_short.shape[0]
    sequence_length = token_dataset_short.shape[1]

    # 3. Load models, etc...
    print("="*50 + " [Loading Model] " + "="*50) # DEBUG
    comparer = ResidAndLn2Comparer()

    # 4. Extract the activations
    extracted_activations: ExtractedActivations = comparer.extract_activations(
        token_dataset_short,
        batch_size=40 # Turns out to work on my machine :)
    )
    print(f"ln2s.shape={extracted_activations.ln2s.shape}")
    print(f"sae_outs_per_k.shape={extracted_activations.sae_outs_per_k.shape}")
    print(f"sae_ins.shape={extracted_activations.sae_ins.shape}")
    print(f"ln2s_saed_per_k.shape={extracted_activations.ln2s_saed_per_k.shape}")

    # 5. Get data we want to showcase
    print("="*50 + " [Flattening + Calculating PCA & Errors] " + "="*50) # DEBUG
    extracted_activations_flattened: FlattenedExtractedActivations = extracted_activations.flatten()

    # 6. Setup FS
    global_plot_folder_path = Path("sae_sans_plots")
    if global_plot_folder_path.exists() and len(list(global_plot_folder_path.glob("*"))) == 0:
        shutil.rmtree(global_plot_folder_path)
    global_plot_folder_path.mkdir(parents=True, exist_ok=False)

    # 7. Plot the error value data
    print("="*50 + " [Plotting Error Value Data] " + "="*50) # DEBUG
    main_helper_plot_error_value_data(
        comparer,
        extracted_activations_flattened,
        global_plot_folder_path
    )

    # 8. Plot the PC cosine sims
    print("="*50 + " [Plotting PC Cosine Sim Data] " + "="*50) # DEBUG
    main_helper_plot_pc_cosine_sims(
        comparer,
        extracted_activations_flattened,
        global_plot_folder_path
    )

    # 9. Plot the PC distribution projections
    print("="*50 + " [Plotting PC Distribution Projections] " + "="*50) # DEBUG
    main_helper_plot_distribution_pc_projections(
        comparer,
        extracted_activations_flattened,
        global_plot_folder_path
    )

    # 10. Plot the PC error projections
    print("="*50 + " [Plotting PC Error Projections] " + "="*50) # DEBUG
    main_helper_plot_errors_pc_projections(
        comparer,
        extracted_activations_flattened,
        global_plot_folder_path
    )
    print("="*50 + " [Done!] " + "="*50) # DEBUG

if __name__ == "__main__":
    main()
