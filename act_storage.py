from __future__ import annotations
"""
A set of barebones utilities to load and save safetensors files for
activations. It is expected that you will provide some kind of big tensor that you want
to store.
"""
from typing import List
import torch
import pydantic
from safetensors.torch import load_file
import tqdm
from safetensors.torch import save_file
from pathlib import Path

# Safetensors key
HIDDEN_STATES_KEY = "hidden_states"

class StoredActivationsMetadata(pydantic.BaseModel):
    per_file_batch_size: int
    total_num_datapoints: int
    file_names: List[str]

def load_files2tensor(subfolder: Path) -> torch.Tensor:
    # Ensure folder/file consistency on a high level
    assert subfolder.exists()
    assert subfolder.is_dir()
    assert next(subfolder.iterdir(), None) is not None
    metadata_file = subfolder / "metadata.json"
    assert metadata_file.exists()
    # Load metadata and ensure correct files present
    meta = StoredActivationsMetadata.model_validate_json(metadata_file.read_text())
    files = [subfolder / f for f in meta.file_names]
    assert all(f.exists() and f.is_file() for f in files)
    assert len(list(subfolder.iterdir())) == len(files) + 1 # +1 for the metadata file

    # Load each file and concatenate
    tensors: List[torch.Tensor] = []
    og_dim = None
    for file in tqdm.tqdm(files, desc="Loading activations", total=len(files)):
        # Load and ensure consistency
        tensors.append(load_file(file)[HIDDEN_STATES_KEY])
        assert og_dim is None or tensors[-1].ndim == og_dim
        og_dim = tensors[-1].ndim
    # Make sure they are cattable by shpae
    non_cat_shape = tensors[0].shape[1:]
    assert all(t.shape[1:] == non_cat_shape for t in tensors)
    # Cat
    catted = torch.cat(tensors, dim=0)
    # Make sure we didn't fk up the shape (i.e. stack etc...)
    # & return
    assert (
        (og_dim is None and meta.total_num_datapoints == 0) or
        (og_dim is not None and catted.ndim == og_dim)
    )
    return catted
        
        

def store_tensor2files(tensor: torch.Tensor, output_subdir: Path, file_batch_size: int) -> None:
    assert output_subdir.exists()
    assert output_subdir.is_dir()
    assert next(output_subdir.iterdir(), None) is None

    ################################
    # First store the Metadata JSON
    ################################
    file_names: List[str] = [f"activations_{i}.safetensors" for i in range(0, tensor.shape[0], file_batch_size)] # fmt: skip
    activations_metadata_file = output_subdir / "metadata.json"
    activations_metadata_file.write_text(
        StoredActivationsMetadata(
            per_file_batch_size=file_batch_size,
            total_num_datapoints=tensor.shape[0],
            file_names=file_names
        ).model_dump_json(indent=4)
    )

    ################################
    # Next, store the activations
    ################################
    for j, i in enumerate(tqdm.trange(0, tensor.shape[0], file_batch_size, desc="Saving activations", total=len(file_names))): # fmt: skip
        # 6K dataset example: Hidden states shape: torch.Size([6000, 17, 101, 2048])
        file_name = output_subdir / file_names[j]
        print("Saving hidden states to:", file_name.as_posix())
        save_file({HIDDEN_STATES_KEY: tensor[i:i+file_batch_size, :, :, :]}, file_name) # fmt: skip