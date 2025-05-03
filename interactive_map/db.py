from __future__ import annotations
"""
(Mostly) Stateless tooling to take in inputs and tag them. This can be used for tagging things as,
say, adjectives vs. nouns, punctuation vs. words, based on frequency, etc...

It is meant to be a modular system that allows you create new tag objects and just slot them into your
"database" for the purposes of visualizing.

We need to have the following capabilities:
- Tag by 
"""
import abc
from jaxtyping import Float, Int, Bool
import torch
import numpy as np
import pydantic
from pathlib import Path
from typing import Any, Optional, Dict, Union, Tuple, List, Callable, Sequence
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm.auto import tqdm
import functools as Ft
import time
import gc
from torch import Tensor
# In the future we may support transformer lens and SAEs from sae-lens
# for analysis but not right now :/
# from sae_utils import SAE
# from transformer_lens.hook_points import HookPoint
from fs import store_tensor2files, load_files2tensor
from transformer_lens.utils import tokenize_and_concatenate
# from transformer_lens import HookedTransformer
import itertools
import datetime
import hashlib


class HookManager:
    """
    A context manager for attaching hooks to a model.

    You pass in a list of tuples of the form (hook_name, is_pre_hook, hook_fn). 

    By Claude with modifications.
    """
    def __init__(
      self,
      model: torch.nn.Module,
      hooks: List[
          Tuple[str, bool, Callable] |
          Tuple[str, Callable]
        ],
      # You can pass one of these two to make ALL of the hooks post/pre hooks
      # You can only pass ONE at most and if you do you should NOT have
      # specified above whether to post or not.
      post: bool = False,
      pre: bool = False,
    ):
        if post and pre:
            raise ValueError("You cannot pass both post and pre hooks")
        if post or pre:
            if any(isinstance(h, tuple) and len(h) == 3 for h in hooks):
                raise ValueError("You cannot pass both a post/pre specific to a hook and a global post/pre specification.") # fmt: skip
            is_pre = pre
            if not is_pre:
                assert post, f"You passed in a global post hook but did not pass in a post hook for {hooks}" # fmt: skip
            hooks = [
                (hook_name, is_pre, hook_fn)
                for hook_name, hook_fn in hooks
            ]
        self.model = model
        self.hooks = hooks
        self.handles = []
        
    def __enter__(self):
        for hook_name, is_pre_hook, hook_fn in self.hooks:
            module = Utils.resolve_module(self.model, hook_name)
            if is_pre_hook:
                handle = module.register_forward_pre_hook(hook_fn)
            else:
                handle = module.register_forward_hook(hook_fn)
            self.handles.append(handle)
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        exceptions_raised = []
        for handle in self.handles:
            try:
                handle.remove()
            except Exception as e:
                exceptions_raised.append(e)
        if exceptions_raised:
            raise Exception(f"Failed to remove {len(exceptions_raised)} hooks: {exceptions_raised}")
        self.handles = []

class Utils:
    # NOTE: later we will implement a dummy model that we will use to debug
    # (it basically returns random tokens/random activations, etc...)
    # SPECIAL_DEBUG_MODEL_NAME = "debug_model"

    """
    A static class to provide utility functions (such
    as for collecting activations, etc...) It supports things such as:
    1. Fetching a model object given the NAME of the model
    2. Fetching a tokenizer for a model given the NAME of the model
    3. Fetching a dataset for a model given the NAME of the dataset (and some
       simple parameters such as size)
    4. Fetching activations for a model given the NAME of the model,
        the LAYER/HOOKPOINT.
    5. Helper methods such as to make hookpoints easy, etc...

    NOTE, for now everything is supposed to be basically on CUDA or CPU in
    a rather hardcoded/simplistic way. This may change in the future but for now...
    eh.

    Everything is meant for INFERENCE so we take all possible steps to
    avoid using gradients, etc... (this may change in the future but not
    for this initial version.)
    """

    def clear_cuda_cache():
        gc.collect()
        torch.cuda.empty_cache()

    ################ HELPERS FOR RUNTIME ################
    @staticmethod
    def resolve_module(mdl: torch.nn.Module, hookpoint: str) -> torch.nn.Module:
        """
        Get the module pointed to by the hookpoint.
        """
        parts = hookpoint.split(".")
        for part in parts:
            if part.isdigit():
                mdl = mdl[int(part)]
            else:
                mdl = getattr(mdl, part)
        return mdl
    
    ################ HELPERS FOR HOOKS ################
    @staticmethod
    def get_gemma_resid_hookname(layer: int):
        """ Hook for the gemma HF model. """
        return f"model.layers.{layer}"
    @staticmethod
    def get_gpt2_resid_hookname(layer: int):
        """ Hook for the gpt2 HF model. """
        return f"transformer.h.{layer}"
    @staticmethod
    def get_gemma2_ln1_hookname(layer: int):
        return f"{Utils.get_gemma_resid_hookname(layer)}.input_layernorm"
    @staticmethod
    def get_gemma2_ln2_hookname(layer: int):
        # TODO(Adriano) not sure on the different between ln2 and resid_post
        return f"{Utils.get_gemma_resid_hookname(layer)}.pre_feedforward_layernorm"
    @staticmethod
    def get_gemma2_resid_hookname(layer: int):
        return f"{Utils.get_gemma_resid_hookname(layer)}.resid_post"
    @staticmethod
    def get_gpt2_ln1_hookname(layer: int):
        return f"{Utils.get_gpt2_resid_hookname(layer)}.ln_1"
    @staticmethod
    def get_gpt2_ln2_hookname(layer: int):
        return f"{Utils.get_gpt2_resid_hookname(layer)}.ln_2"

    @staticmethod
    def with_hooks(
        model: torch.nn.Module,
        hooks: List[
            Tuple[str, bool, Callable] |
            Tuple[str, Callable]
        ],
        post: bool = False,
        pre: bool = False,
      ):
        """
        Context manager to temporarily attach hooks to a model. It is
        a wrapper around the HookManager class.
        """
        return HookManager(model, hooks, post, pre)
    
    @staticmethod
    def collect_activations_post_hook_fn(
        collection_list: list[torch.Tensor],
        mod, 
        inputs, 
        outputs,
    ):
        """ Hook function to collect activations. """
        collection_list.append(outputs[0].detach().requires_grad_(False).cpu())
        return outputs

    ################ RUNTIME UTILITY FUNCTIONS ################
    @staticmethod
    def get_model(model_name: str = "google/gemma-2-2b"):
        """Get a model object given the NAME of the model."""
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cuda")
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
            p.grad = None
        return model
    
    @staticmethod
    def get_tokenizer(model_name: str = "google/gemma-2-2b"):
        return AutoTokenizer.from_pretrained(model_name)

    @staticmethod
    def get_dataset(
            dataset_name: str = "stas/openwebtext-10k",
            context_length: int = 128,
            batch_size: Optional[int] = None,
            tokenizer: Optional[AutoTokenizer] = None,            
        ):
        if tokenizer is None:
            raise ValueError("Tokenizer is required for `get_dataset`")
        dataset = load_dataset(
            dataset_name,
            split="train",
            trust_remote_code=True,
        )
        tokens = tokenize_and_concatenate(
            dataset=dataset,
            tokenizer=tokenizer,
            streaming=True,
            max_length=context_length,
            add_bos_token=True,
        )["tokens"]
        tokens = tokens.cpu()
        if batch_size is not None:
            tokens = tokens[:batch_size]
        return tokens

    @staticmethod
    def get_activations(
            model_name: str | torch.nn.Module = "google/gemma-2-2b",
            tokenizer: Optional[AutoTokenizer] = None,
            # Hook point can be a hook point STRING or a layer
            hookpoint: Optional[str] = None,
            layer: Optional[int] = None,
            hookpoint_alias: Optional[str] = None,
            # Whether to return the raw activations or not
            return_raw: bool = False,
            # Size of the dataset/other parameters
            # (if you do not provide a string it is assumed to be the
            # TOKENS themselves)
            dataset: str | Any = "stas/openwebtext-10k",
            batch_size: int = 10_000,
            context_length: int = 500,
            # Inference batch size is used 
            inference_batch_size: int = 100,
        ) -> Tuple[
            # Activations
            Float[torch.Tensor, "batch token"],
            # Tokens
            Int[torch.Tensor, "batch"],
            # Special tokens
            Bool[torch.Tensor, "batch"],
            Bool[torch.Tensor, "batch"],
            Bool[torch.Tensor, "batch"],
            Bool[torch.Tensor, "batch"],
        ]:

        ########## 0. Set up the hookpoint name ##########
        # You can provide EITHER
        # 1. The hookpoint
        # 2. The layer ONLY (get resid)
        # 3. The hookpoint alias AND the layer (get resid or ln1, ln2, etc...)
        #    (note that "resid" is the alias for the resid hookpoint and 
        #    "ln*" is the alias for the ln* hookpoint)
        provided_args = [hookpoint, layer, hookpoint_alias]
        if (
            hookpoint is not None and
            layer is not None and
            hookpoint_alias is not None
        ):
            raise ValueError("You cannot provide both a hookpoint and a layer/hookpoint_alias")
        elif hookpoint is not None:
            if layer is not None or hookpoint_alias is not None:
                raise ValueError("You cannot provide a hookpoint and a layer/hookpoint_alias")
        elif hookpoint_alias is not None:
            if layer is not None:
                raise ValueError("You cannot provide a hookpoint_alias and a layer")
        
        if hookpoint is not None:
            hookpoint_name = hookpoint
        else:
            if hookpoint_alias is None:
                hookpoint_alias = "resid"
            # Get resids
            if model_name == "google/gemma-2-2b" and hookpoint_alias == "resid":
                hookpoint_name = Utils.get_gemma_resid_hookname(layer)
            elif model_name == "gpt2" and hookpoint_alias == "resid":
                hookpoint_name = Utils.get_gpt2_resid_hookname(layer)
            # Get ln1s
            elif model_name == "google/gemma-2-2b" and hookpoint_alias == "ln1":
                hookpoint_name = Utils.get_gemma2_ln1_hookname(layer)
            elif model_name == "gpt2" and hookpoint_alias == "ln1":
                hookpoint_name = Utils.get_gpt2_ln1_hookname(layer)
            # Get ln2s
            elif model_name == "google/gemma-2-2b" and hookpoint_alias == "ln2":
                hookpoint_name = Utils.get_gemma2_ln2_hookname(layer)
            elif model_name == "gpt2" and hookpoint_alias == "ln2":
                hookpoint_name = Utils.get_gpt2_ln2_hookname(layer)
            else:
                raise ValueError(f"Model {model_name} not supported for hookpoint name aliasing")
            
        torch.set_grad_enabled(False)
        with torch.no_grad():
            ########## 1. Get our model tokenizer etc... ##########
            model = model_name
            if isinstance(model_name, str):
                model = Utils.get_model(model_name)
                if tokenizer is None:
                    tokenizer = Utils.get_tokenizer(model_name)
            elif tokenizer is None:
                try:
                    tokenizer = model.tokenizer
                except AttributeError:
                    raise ValueError("Tokenizer is required for `get_activations` if you don't provide a model name and it's not obviously accessible.") # fmt: skip

            ########## 2. Get the full dataset ##########
            tokens = dataset
            if isinstance(dataset, str):
                tokens = Utils.get_dataset(
                    dataset_name=dataset,
                    batch_size=batch_size,
                    tokenizer=tokenizer,
                    # Context setup information
                    context_length=context_length,
                )
            elif isinstance(dataset, torch.Tensor):
                assert tokens.ndim == 2, f"Expected a 2D tensor, got {tokens.shape}"
                tokens = tokens[:batch_size, :context_length]
            else:
                raise ValueError(f"Invalid dataset type: {type(dataset)}")
            try:
                model_device = model.device
            except AttributeError:
                model_device = next(model.parameters()).device
            tokens = tokens.to(model_device)

        #### 3. Collect Activations ####
        with torch.no_grad():
            collected_outputs = []
            collect_hook_fn = Ft.partial(Utils.collect_activations_post_hook_fn, collection_list=collected_outputs)
            with Utils.with_hooks(
                model,
                [
                    (hookpoint_name, False, collect_hook_fn),
                ],
            ):
                for i in tqdm.trange(0, tokens.shape[0], inference_batch_size, desc=f"Forward pass @ inference batch size={inference_batch_size}"): # fmt: skip
                    j = min(i + inference_batch_size, tokens.shape[0])
                    model.forward(tokens[i:j])
            collected_outputs = torch.cat(collected_outputs, dim=0)
            # print(collected_outputs.shape) # DEBUG

        tokens_is_eos = (tokens == tokenizer.eos_token_id).cpu()
        tokens_is_bos = (tokens == tokenizer.bos_token_id).cpu()
        tokens_is_pad = (tokens == tokenizer.pad_token_id).cpu()
        tokens_is_special = tokens_is_eos | tokens_is_bos | tokens_is_pad
        tokens = tokens.cpu()
        assert collected_outputs.ndim == 3, f"collected_outputs.shape: {collected_outputs.shape}"
        assert collected_outputs.shape[:-1] == tokens.shape, f"collected_outputs.shape: {collected_outputs.shape}, tokens.shape: {tokens.shape}" # fmt: skip
        assert tokens_is_eos.shape == tokens_is_bos.shape == tokens_is_pad.shape == tokens.shape, f"tokens_is_eos.shape: {tokens_is_eos.shape}, tokens_is_bos.shape: {tokens_is_bos.shape}, tokens_is_pad.shape: {tokens_is_pad.shape}, tokens.shape: {tokens.shape}" # fmt: skip
        return (
            # Activations
            collected_outputs,
            # Tokens
            tokens,
            # Special tokens TAGS contents
            tokens_is_special,
            tokens_is_eos,
            tokens_is_bos,
        )

class ActivationsDB(pydantic.BaseModel):
    """
    By Claude with modifications.

    A database state machine for activations that you might want to analyze. Basically
    you can do the following (rather independently):
    
    Modify the state:
    - Add tags, remove tags, modify tags
        (to existing activations)
    - Add new activations, remove activations, modify activations
        (possibly for new hookpoints, layers, etc...)
    - Define new viewports (the projection matrices we use to get the
        2D histograms, usually done via PCA) This is now done in the following
        ways:
            1. Provide a 2D subspace via a matrix.
            2. Provide a dataset of activations and a filtering criteria
                (based on the tags) to select a subset. That subset of the
                activations will then be used to compute a PCA.
            3. Train one linear/logistic classifier on a boolean tag and get
                a random 2nd axis.
            4. Train two linear/logistic classifiers on one or
                two boolean tags.
        You can:
            - Save viewports
            - Load/use saved viewports (and view which are available)
                (the proper way to do this is to give them an informative
                name/ID and a description)
    
    Fetch visualizeables:
    - Return 2D arrays of histos that we can visualize
        (numpy arrays) that represent the orthographic
        projections onto those subspaces.
    
        
    Exploit persistence:
    - Load and save from a folder in a "stateless" way (that is to say
        you can load it again in different runtime, different machine,
        etc... and in principle it should just work)
    
    TODO(Adriano) in the future you will 
    """
    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    ################ Basic Configuration ################
    # Basic metadata
    db_instance_name: str
    version: str = "1.0.0"
    root_directory: Path
    
    # Configuration
    verbose: bool = True
    device: torch.device = pydantic.Field(
        default_factory=lambda: torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    
    # Model information
    model_name: str
    hooks: list[str] = pydantic.Field(default_factory=list)
    
    # These will be initialized but aren't part of the Pydantic model
    tokenizer: Optional[Any] = pydantic.Field(default=None, exclude=True)
    model: Optional[Any] = pydantic.Field(default=None, exclude=True)
    
    # Internal state - activation cache
    _activations_cache: Dict[str, torch.Tensor] = pydantic.Field(default_factory=dict, exclude=True)
    _tokens_cache: Dict[str, torch.Tensor] = pydantic.Field(default_factory=dict, exclude=True)
    
    # Cache for tokens with tags
    _tag_tokens: Dict[str, torch.Tensor] = pydantic.Field(default_factory=dict, exclude=True)
    
    # Cache for viewport projections
    _viewports: Dict[str, Dict[str, Any]] = pydantic.Field(default_factory=dict, exclude=True)
    
    # ════════════════ Initialization ════════════════
    def __init__(self, **data):
        super().__init__(**data)
        # Convert root_directory to Path if it's a string
        if isinstance(self.root_directory, str):
            self.root_directory = Path(self.root_directory)
        
        # Create directory structure if it doesn't exist
        self._setup_directories()
        
        # Initialize tokenizer and model if not provided
        if self.tokenizer is None and self.model_name:
            self.tokenizer = Utils.get_tokenizer(self.model_name)
        
        # Load existing tags
        self._load_tags()
    
    def _setup_directories(self):
        """Create necessary directories for the database."""
        # Main directory
        self.root_directory.mkdir(parents=True, exist_ok=True)
        
        # Subdirectories
        (self.root_directory / "activations").mkdir(exist_ok=True)
        (self.root_directory / "tags").mkdir(exist_ok=True)
        (self.root_directory / "viewports").mkdir(exist_ok=True)
        
        # Save metadata
        # TODO(Claude) please create a pydantic model for ActivationsDBMMetadata
        meta = {
            "db_instance_name": self.db_instance_name,
            "version": self.version,
            "model_name": self.model_name,
            "hooks": self.hooks,
            "creation_date": datetime.datetime.now().isoformat(),
        }
        
        meta_file = self.root_directory / "meta.json"
        if not meta_file.exists():
            with open(meta_file, "w") as f:
                json.dump(meta, f, indent=2)
    
    def _load_tags(self):
        """Load existing tags from disk."""
        tags_dir = self.root_directory / "tags"
        if tags_dir.exists():
            # TODO(Claude) please NEVER use pt and instead prefer to use safetensors
            # I have provided some utilities in `fs.py` that are imported above; they basically
            # store a tensor to a directory; this will mean that instead of storing to files
            # you will be storing to directories; there is no real big difference, but
            # you can just know it in case it is relevant to your use-case and implementation
            # (if you want to store a single file: recommended for tokens and masks, make sure
            # to set a batch_size that is really big, i.e. bigger than the dataset's size)
            # Read the code to know what to do, Claude (`fs.py`)
            for tag_file in tags_dir.glob("tag_*.pt"):
                tag_name = tag_file.stem[4:]  # Remove 'tag_' prefix
                try:
                    self._tag_tokens[tag_name] = torch.load(tag_file).to(self.device)
                    if self.verbose:
                        print(f"Loaded tag '{tag_name}' with {len(self._tag_tokens[tag_name])} tokens")
                except Exception as e:
                    print(f"Error loading tag '{tag_name}': {e}")
    
    # ════════════════ Factory Methods ════════════════
    @classmethod
    def build(cls, 
              model_name: str,
              hooks: list[str],
              db_instance_name: Optional[str] = None,
              run_dir: Optional[str] = None,
              verbose: bool = True):
        """
        Create a new ActivationsDB instance.
        
        Args:
            model_name: Name of the model to use
            hooks: List of hook points to capture
            db_instance_name: Optional name for the database (defaults to model name)
            run_dir: Optional directory to store the database (defaults to 'runs/{model_name}_{db_hash}')
            verbose: Whether to show verbose output
        
        Returns:
            ActivationsDB instance
        """
        # Generate a default database name if not provided
        now = datetime.datetime.now()
        if db_name is None:
            db_name = f"{model_name.split('/')[-1]}_db_{now.strftime('%Y_%m_%d_%H_%M_%S')}"
        
        # Create a hash for the configuration
        config = {
            "model_name": model_name,
            "hooks": hooks,
            "db_name": db_name,
            "creation_date": now.isoformat(),
        }
        
        # Generate a hash from the configuration
        config_hash = hashlib.sha1(json.dumps(config, sort_keys=True).encode()).hexdigest()[:8]
        
        # Determine run directory
        if run_dir is None:
            subdir_name = model_name.replace("/", "_")
            run_dir = f"runs/{subdir_name}_{config_hash}"
        
        # Create the instance
        instance = cls(
            db_name=db_name,
            model_name=model_name,
            hooks=hooks,
            root_directory=Path(run_dir),
            verbose=verbose
        )
        
        return instance
    
    @classmethod
    def from_path(cls, path: str | Path) -> 'ActivationsDB':
        """
        Load an ActivationsDB from a directory.
        
        Args:
            path: Path to the directory containing the database
            
        Returns:
            ActivationsDB instance
        """
        path = Path(path)
        
        # Load metadata
        meta_path = path / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"No metadata found at {meta_path}")
        
        with open(meta_path) as f:
            meta = json.load(f)
        
        # Create the instance
        instance = cls(
            db_name=meta.get("db_name", path.name),
            model_name=meta.get("model_name", "unknown"),
            hooks=meta.get("hooks", []),
            version=meta.get("version", "1.0.0"),
            root_directory=path,
            verbose=True
        )
        
        return instance
    
    # ════════════════ Activation Management ════════════════
    def get_activations(self, 
                       hook: str,
                       dataset: str | torch.Tensor = "stas/openwebtext-10k",
                       batch_size: int = 1000,
                       context_length: int = 128,
                       force_recompute: bool = False) -> torch.Tensor:
        """
        Get activations for a specific hook point.
        
        If activations are already cached, returns them.
        Otherwise, generates new activations and caches them.
        
        Args:
            hook: Hook point to get activations for
            dataset: Dataset to use for generating activations
            batch_size: Batch size for activation generation
            context_length: Context length for tokens
            force_recompute: Whether to force recomputation even if cache exists
            
        Returns:
            Activation tensor
        """
        # TODO(Claude) please create utility methods for creating your
        # cache keys, etc... you should make your code self-documenting, modular
        # easy to build on, etc... (the less comments you NEED the better)
        #
        #
        # Create a cache key based on hook and dataset
        if isinstance(dataset, str):
            cache_key = f"{hook}_{dataset}_{batch_size}_{context_length}"
        else:
            # For tensor datasets, use a hash of the first few tokens
            sample = dataset[:min(10, dataset.shape[0]), :min(10, dataset.shape[1])].flatten()
            cache_key = f"{hook}_{hash(tuple(sample.tolist()))}_{batch_size}_{context_length}"
        
        # Check if activations are already cached
        if not force_recompute and cache_key in self._activations_cache:
            if self.verbose:
                print(f"Using cached activations for {hook}")
            return self._activations_cache[cache_key]
        
        # Load the model if not already loaded
        if self.model is None:
            if self.verbose:
                print(f"Loading model {self.model_name}")
            self.model = Utils.get_model(self.model_name)
        
        # Generate activations
        if self.verbose:
            print(f"Generating activations for {hook}")
        
        # Determine the actual hook point name if needed
        if "." not in hook and hook.isdigit():
            # Assuming it's a layer number
            layer = int(hook)
            hookpoint = None
        else:
            # Assuming it's a full hook point name
            layer = None
            hookpoint = hook
        
        # Generate activations using Utils
        activations, tokens, _, _, _ = Utils.get_activations(
            model_name=self.model,
            tokenizer=self.tokenizer,
            hookpoint=hookpoint,
            layer=layer,
            dataset=dataset,
            batch_size=batch_size,
            context_length=context_length
        )
        
        # Cache the results
        self._activations_cache[cache_key] = activations
        self._tokens_cache[cache_key] = tokens
        
        # Save to disk
        # TODO(Claude) please replace this to use my utility from above which
        # I briefly describe when you use pt before this
        act_path = self.root_directory / "activations" / f"{cache_key}.pt"
        torch.save(activations.cpu(), act_path)
        
        tokens_path = self.root_directory / "activations" / f"{cache_key}_tokens.pt"
        torch.save(tokens.cpu(), tokens_path)
        
        return activations
    
    def list_cached_activations(self) -> list[str]:
        """List all cached activations."""
        return sorted(list(self._activations_cache.keys()))
    
    ################ Tag Management ################
    def add_tag(self, name: str, 
                predicate: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                hook: str,
                dataset: str | torch.Tensor = "stas/openwebtext-10k",
                batch_size: int = 1000,
                context_length: int = 128):
        """
        Add a tag to the database based on a predicate.
        
        Args:
            name: Tag name
            predicate: Function that takes (activations, tokens) and returns a boolean mask
            hook: Hook point to apply the predicate to
            dataset: Dataset to use for generating activations
            batch_size: Batch size for activation generation
            context_length: Context length for tokens
        """
        # Get activations for the hook
        activations = self.get_activations(
            hook=hook,
            dataset=dataset,
            batch_size=batch_size,
            context_length=context_length
        )
        
        # Get tokens for the same hook
        cache_key = f"{hook}_{dataset}_{batch_size}_{context_length}"
        tokens = self._tokens_cache.get(cache_key)
        
        if tokens is None:
            raise ValueError(f"No tokens found for {hook}. This should not happen if activations exist.")
        
        # Apply predicate to get mask
        mask = predicate(activations, tokens)
        
        if not isinstance(mask, torch.Tensor):
            mask = torch.tensor(mask, device=self.device)
        
        if mask.dtype != torch.bool:
            mask = mask.bool()
        
        # Get token indices where mask is True
        tagged_indices = torch.nonzero(mask.flatten(), as_tuple=False).flatten()
        
        if len(tagged_indices) == 0:
            if self.verbose:
                print(f"No tokens matched the predicate for tag '{name}'")
            return
        
        # Store the tag
        self._tag_tokens[name] = tagged_indices
        
        # Save to disk
        tag_path = self.root_directory / "tags" / f"tag_{name}.pt"
        torch.save(tagged_indices.cpu(), tag_path)
        
        if self.verbose:
            print(f"Added tag '{name}' with {len(tagged_indices)} tokens")
    
    def remove_tag(self, name: str):
        """
        Remove a tag from the database.
        
        Args:
            name: Tag name to remove
        """
        if name not in self._tag_tokens:
            if self.verbose:
                print(f"Tag '{name}' not found")
            return
        
        # Remove from memory
        del self._tag_tokens[name]
        
        # Remove from disk
        tag_path = self.root_directory / "tags" / f"tag_{name}.pt"
        if tag_path.exists():
            tag_path.unlink()
        
        if self.verbose:
            print(f"Removed tag '{name}'")
        
        Utils.clear_cuda_cache()
    
    def count(self, tag: str, *, hook: Optional[str] = None) -> int:
        """
        Count tokens with a specific tag, optionally filtered by hook.
        
        Args:
            tag: Tag name to count
            hook: Optional hook to filter by
            
        Returns:
            Number of tokens with the tag
        """
        if tag not in self._tag_tokens:
            return 0
        
        # If no hook filter, return all tokens with the tag
        if hook is None:
            return len(self._tag_tokens[tag])
        
        # Otherwise, count tokens for the specific hook
        # This would require hook-specific tracking which we don't have yet
        # For now, just return the total number of tagged tokens
        return len(self._tag_tokens[tag])
    
    def list_tags(self) -> list[str]:
        """List all available tags."""
        return list(self._tag_tokens.keys())
    
    # ════════════════ Visualization Support ════════════════
    def define_viewport(self, name: str, 
                       description: str, 
                       matrix: Optional[np.ndarray] = None,
                       hook: Optional[str] = None,
                       tag_filter: Optional[str] = None,
                       classifier_tags: Optional[list[str]] = None):
        """
        Define a viewport (projection matrix) for visualization.
        
        Args:
            name: Viewport name
            description: Viewport description
            matrix: Optional 2D projection matrix (will be computed if not provided)
            hook: Hook point to use for PCA if matrix not provided
            tag_filter: Optional tag to filter activations for PCA
            classifier_tags: Optional tags to use for classifier-based projection
        """
        raise NotImplementedError("Viewport methods not yet implemented")
    
    def list_viewports(self) -> list[dict]:
        """
        List all available viewports.
        
        Returns:
            List of viewport metadata dictionaries
        """
        raise NotImplementedError("Viewport methods not yet implemented")
    
    def get_projection(self, viewport_name: str, tag_filter: Optional[str] = None) -> np.ndarray:
        """
        Get projected activations for visualization.
        
        Args:
            viewport_name: Name of the viewport to use
            tag_filter: Optional tag to filter activations
            
        Returns:
            2D numpy array of projected activations
        """
        raise NotImplementedError("Viewport methods not yet implemented")
    
    # ════════════════ Utility Methods ════════════════
    def random_windows(self, tag: str, k: int, radius: int = 8, color: bool = True) -> list[tuple[str, int]]:
        """
        Get random windows of tokens around tagged tokens.
        
        Args:
            tag: Tag name to select windows
            k: Number of windows to return
            radius: Window radius in tokens
            color: Whether to highlight the trigger token
            
        Returns:
            List of (window_text, trigger_id) tuples
        """
        if tag not in self._tag_tokens:
            if self.verbose:
                print(f"Tag '{tag}' not found")
            return []
        
        # Get a random sample of tagged tokens
        tagged_indices = self._tag_tokens[tag]
        k = min(k, len(tagged_indices))
        
        if k == 0:
            return []
        
        # Select random indices
        from random import sample
        selected_indices = sample(tagged_indices.tolist(), k)
        
        # Find which cached activations each index belongs to
        windows = []
        
        # This is a simplified implementation that would need to be expanded
        # based on how we map token indices to actual tokens in sequences
        # For now, just return placeholder data
        return [("Sample window with [HIGHLIGHTED] token", 0) for _ in range(k)]
    
    # ════════════════ Persistence ════════════════
    def sync(self, to_path: Optional[Path] = None):
        """
        Save the database state to disk.
        
        Args:
            to_path: Optional path to save to (uses root_directory if not provided)
        """
        target_path = to_path if to_path is not None else self.root_directory
        target_path = Path(target_path)
        
        # Ensure directories exist
        target_path.mkdir(parents=True, exist_ok=True)
        (target_path / "activations").mkdir(exist_ok=True)
        (target_path / "tags").mkdir(exist_ok=True)
        (target_path / "viewports").mkdir(exist_ok=True)
        
        # Save metadata
        meta = {
            "db_instance_name": self.db_instance_name,
            "version": self.version,
            "model_name": self.model_name,
            "hooks": self.hooks,
            "last_updated": str(datetime.datetime.now()),
        }
        
        with open(target_path / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        
        # Save all cached activations
        for key, acts in self._activations_cache.items():
            torch.save(acts.cpu(), target_path / "activations" / f"{key}.pt")
        
        # Save all tokens
        for key, tokens in self._tokens_cache.items():
            torch.save(tokens.cpu(), target_path / "activations" / f"{key}_tokens.pt")
        
        # Save all tags
        for tag_name, indices in self._tag_tokens.items():
            torch.save(indices.cpu(), target_path / "tags" / f"tag_{tag_name}.pt")
        
        if self.verbose:
            print(f"Database synced to {target_path}")

