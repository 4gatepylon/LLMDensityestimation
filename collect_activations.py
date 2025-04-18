from __future__ import annotations
from pathlib import Path
import transformer_lens
import torch
import tqdm
import math
import einops
import dotenv
from typing import List, Tuple
import functools as Ft
from transformer_lens.hook_points import HookPoint
from transformer_lens import HookedTransformer
from act_storage import store_tensor2files
from datasets import load_dataset
dotenv.load_dotenv() # Should constrain cuda device if necessary
device = "cuda"

"""
Utility functions that we use to basically collect activations, etc...
"""

def collect_activations_hook(
        # Input
        activation: torch.Tensor,
        # HTF thing
        hook: HookPoint,
        # We will be saving to a tensor
        activations_save_loc: torch.Tensor,
         # Pass these below for sanity t esting
        left: int,
        right: int,
        model_dim: int,
        seq_len: int,
        hook_idx: int,
    ):
    """
    A simple hook function to used in `collect_all_activations` to save a buncha activations.
    """
    # Sanity check activations
    assert activation.ndim == 3, f"activation.ndim = {activation.ndim}" # fmt: skip
    assert activation.shape[0] == right - left, f"activation.shape[0] = {activation.shape[0]}, right - left = {right - left}" # fmt: skip
    assert activation.shape[1] == seq_len, f"activation.shape[1] = {activation.shape[1]}, model_seq_size = {seq_len}" # fmt: skip
    assert activation.shape[2] == model_dim, f"activation.shape[2] = {activation.shape[2]}, model_dim = {model_dim}" # fmt: skip

    # Sanity check outputs
    assert activations_save_loc.ndim == 4
    assert 0 <= hook_idx < activations_save_loc.shape[0]
    assert 0 <= left < right <= activations_save_loc.shape[1]
    assert 0 <= activation.shape[1] == activations_save_loc.shape[2]
    assert 0 <= activation.shape[2] == activations_save_loc.shape[3]

    # Store SAE IO
    activations_save_loc[hook_idx, left:right, :, :] = activation.detach().cpu()
    return activation


@torch.no_grad()
def collect_all_activations(
        model: HookedTransformer,
        inputs: torch.Tensor,
        hook_names: List[str],
        inference_batch_size: int = 20,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    TODO(Adriano): support a boolean mask to exclude EOS.
    """
    assert isinstance(hook_names, list), f"hook_names is not a list: {hook_names}" # fmt: skip
    assert all([isinstance(hook_name, str) for hook_name in hook_names]), f"hook_names is not a list of strings: {hook_names}" # fmt: skip
    assert isinstance(inputs, torch.Tensor), f"inputs is not a torch.Tensor: {inputs}" # fmt: skip
    assert inputs.ndim == 2, f"inputs.ndim = {inputs.ndim}" # fmt: skip
    assert isinstance(inference_batch_size, int), f"inference_batch_size is not an int: {inference_batch_size}" # fmt: skip
    assert inference_batch_size > 0, f"inference_batch_size is not positive: {inference_batch_size}" # fmt: skip
    model_training = model.training
    model.eval()
    try:
        assert inputs.ndim == 2
        total_batch_size, seq_len = inputs.shape
        model_dim = model.cfg.d_model
        rnd_sans = (torch.randn(1) * 9999999).item()
        outputs = rnd_sans * torch.ones(len(hook_names), total_batch_size, seq_len, model_dim)
        losses = torch.zeros(math.ceil(total_batch_size / inference_batch_size),)
        pbar = tqdm.trange(0, inputs.shape[0], inference_batch_size)
        for i, left in enumerate(pbar):
            right = min(left+inference_batch_size, total_batch_size)
            assert right > left and right <= total_batch_size
            loss = model.run_with_hooks(
                inputs[left:right],
                # fwd_hooks=[],
                fwd_hooks=[
                        (
                            hook_name,
                            Ft.partial(
                                collect_activations_hook,
                                activations_save_loc=outputs,
                                left=left,
                                right=right,
                                model_dim=model_dim,
                                seq_len=seq_len,
                                hook_idx=h,
                            ),
                        ) for h, hook_name in enumerate(hook_names)
                ],
                return_type="loss",
            )
            losses[i] = loss.item()
            pbar.set_description(f"Average loss is: {torch.mean(losses[:i+1]).item()}")
        assert torch.all(outputs != rnd_sans).item(), f"Some torch outputs ({(outputs == rnd_sans).sum()}/{outputs.numel()}) appear to equal rnd_sans" # fmt: skip
        return outputs, losses
    finally:
        model.train(model_training)

def main():
    model = HookedTransformer.from_pretrained("gpt2-small", device=device)
    dataset = load_dataset("Skylion007/openwebtext", split="train")
    batch_size = 1024 * 256 # NOTE: 1M * 1K => 1B bytes => 1GB ??? that's kind of big lol
    max_amount = 1024 * 1024 * 1024 * 1024 + 1 # 32 million roughly, times 1K => 32GB should be a fine amount tbh
    save_every = 16 # NOTE: this is every this many BATCHES
    file_batch_size = batch_size # eh...
    save_every_counter = 0
    output_folder_name = Path("gpt2_small_activations")
    output_folder_activations = output_folder_name / "activations"
    output_folder_losses = output_folder_name / "losses"
    output_folder_activations.mkdir(parents=True, exist_ok=True)
    output_folder_losses.mkdir(parents=True, exist_ok=True)
    # ...
    hook_names = ["blocks.6.hook_resid_pre"]
    # ...
    full_activations = []
    full_losses = []
    print("Length of dataset:", len(dataset))
    print("Batch size:", batch_size)
    print("Num batches:", math.ceil(len(dataset) / batch_size))
    for i in tqdm.trange(0, len(dataset), batch_size):
        #### Tokenize, etc... ####
        if i > max_amount:
            break # NOTE: this will happen on a power of 2 so we offset by 1 above to save the batch :P; fmt: skip
        batch = dataset[i:i+batch_size]
        assert all(isinstance(x, str) for x in batch), f"Batch is not a list of strings: {batch}" # fmt: skip; fmt: skip
        # texts = [ex["text"] for ex in batch]
        tokens = model.to_tokens(batch)
        assert tokens.ndim == 2, f"tokens.ndim = {tokens.ndim}"
        assert tokens.shape[0] <= batch_size, f"tokens.shape[0] = {tokens.shape[0]}, batch_size = {batch_size}"
        assert tokens.shape[1] <= model.cfg.n_ctx, f"tokens.shape[1] = {tokens.shape[1]}, model.cfg.n_ctx = {model.cfg.n_ctx}"

        #### Collect activations ####
        with torch.no_grad():
            activations, losses = collect_all_activations(model, tokens, hook_names)
            full_activations.append(activations)
            full_losses.append(losses)
        
        #### Save every so often ####
        if save_every_counter >= save_every or i >= max_amount or i >= len(dataset):
            print("="*50 + " [SAVING] " + "="*50)
            save_every_counter = 0
            # TODO(Adriano) fix this shit to be more general purpose lmao
            # full_activations_pt = einops.rearrange(torch.cat(full_activations, dim=0), "b d -> b x y d", x=1, y=1)
            full_activations_pt = torch.cat(full_activations, dim=0)
            assert full_activations_pt.ndim == 4, f"full_activations_pt.shape = {full_activations_pt.shape}"
            full_losses_pt = einops.rearrange(torch.cat(full_losses, dim=0), "b -> b 1 1 1")
            assert full_losses_pt.ndim == 4, f"full_losses_pt.shape = {full_losses_pt.shape}"
            this_batch_name = f"{i:020d}-idx-{i:020d}"
            output_subdir_acts = output_folder_activations / this_batch_name
            output_subdir_losses = output_folder_losses / this_batch_name
            output_subdir_acts.mkdir(parents=True, exist_ok=False) # NOTE: these two should not already exist
            output_subdir_losses.mkdir(parents=True, exist_ok=False)
            # store_tensor2files(full_activations_pt, output_subdir_acts, file_batch_size=file_batch_size)
            # store_tensor2files(full_losses_pt, output_subdir_losses, file_batch_size=file_batch_size)
            full_activations = []
            full_losses = []
            # NOTE: if you broke you will lose the last batch ay lmao
        else:
            save_every_counter += 1

if __name__ == "__main__":
    main()

