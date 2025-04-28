from safetensors.torch import load_file
from safetensors.torch import save_file
import json
from pathlib import Path
from typing import Optional, Callable, Any

def load_checkpoint(checkpoint_path: str | Path, infer_hyperparameters: Callable[[Any], dict]):
    checkpoint_path = Path(checkpoint_path).resolve().as_posix()
    assert isinstance(checkpoint_path, str)
    assert checkpoint_path.count(".") <= 1, "Checkpoint path contains more than one period"
    if checkpoint_path.contains("."):
        print("WARNING: found extension, will remove and try again")
        checkpoint_path = checkpoint_path.rsplit(".", 1)[0]
    safetensors_path = checkpoint_path + ".safetensors"
    hyperparameters_path = checkpoint_path + "_hyperparameters.json"
    if not safetensors_path.exists():
        raise FileNotFoundError(f"No safetensors file found at {safetensors_path}")
    state_dict = load_file(safetensors_path)
    hyperparameters = (
        json.loads(Path(hyperparameters_path).read_text()) if hyperparameters_path.exists() else None
    )
    assert hyperparameters is None or isinstance(hyperparameters, dict)
    if hyperparameters is None:
        print("No hyperparameters really found, inferring from state_dict...")
        hyperparameters = infer_hyperparameters(state_dict)
    assert isinstance(hyperparameters, dict)
    return hyperparameters, state_dict

def save_checkpoint(checkpoint_path: str | Path, hyperparameters: Optional[dict], state_dict: dict, clobber: bool = True):
    # By Claude
    checkpoint_path = Path(checkpoint_path).resolve().as_posix()
    assert isinstance(checkpoint_path, str)
    assert checkpoint_path.count(".") <= 1, "Checkpoint path contains more than one period"
    if checkpoint_path.contains("."):
        print("WARNING: found extension, will remove and try again")
        checkpoint_path = checkpoint_path.rsplit(".", 1)[0]
    safetensors_path = checkpoint_path + ".safetensors"
    hyperparameters_path = checkpoint_path + "_hyperparameters.json"
    if not clobber:
        assert not safetensors_path.exists(), "safetensors file already exists"
        assert not hyperparameters_path.exists(), "hyperparameters file already exists"
    save_file(state_dict, safetensors_path)
    if hyperparameters is not None:
        Path(hyperparameters_path).write_text(json.dumps(hyperparameters, indent=4))