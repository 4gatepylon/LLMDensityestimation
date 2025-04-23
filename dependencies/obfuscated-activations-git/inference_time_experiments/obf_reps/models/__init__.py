from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Optional, Tuple, Type, TypeVar, Union

import torch
import torch.nn.functional as F
from jaxtyping import Bool, Float, Int64
from torch import Tensor, nn
from torch.nn import Embedding
from transformers import PreTrainedTokenizer


class ParamsBase(ABC):
    """Abstract base class for tunable parameters in language models.

    This class defines the interface for parameter types that can be optimized during training.
    Models contain ParamBase as an attribute and they are optimized by Optimizers.
    """

    def reinit(self):
        """Reinitialize parameters to their initial values."""
        ...


class HardParams(ParamsBase):
    """Hard token parameters for discrete token optimization.

    This class stores hard tokens that are optimized and
    inserted into (generally as suffix) to model inputs.
    To make these params interact nicely with discrete
    optimizers we store the tokens as one-hot vectors with
    length equal to the model's vocabulary size.

    Args:
        init_ids: Initial token IDs to optimize from
        embedding: Embedding layer of the model
    """

    def __init__(self, init_ids: Float[Tensor, "1 num_toks"], embedding: Embedding):

        optim_ids_onehot: Float[Tensor, "1 num_toks vocab_size"] = F.one_hot(
            init_ids, num_classes=embedding.num_embeddings
        )

        self.params = optim_ids_onehot.to(
            dtype=embedding.weight.dtype, device=embedding.weight.device
        ).requires_grad_()
        self.init_params = self.params.clone().detach()

    def reinit(self):
        self.params = self.init_params.clone().detach().requires_grad_()

    @property
    def optim_ids(self) -> Float[Tensor, "1 seq_len"]:
        """Get the token IDs of the one hot encoded self.params."""
        return torch.argmax(self.params, dim=-1)


class SoftParams(ParamsBase):
    """Continuous embedding parameters for soft prompt tuning.

    This class stores soft embeddings that are optimized and
    inserted into (generally as suffix) to model inputs
    at the embedding layer.

    Args:
        init_ids: Initial token IDs to derive embeddings from
        embedding: Embedding layer of the model
    """

    def __init__(self, init_ids: Float[Tensor, "1 num_toks"], embedding: Embedding):

        optim_embeds = embedding(init_ids).detach().clone().requires_grad_()

        self.init_params = optim_embeds.clone().detach()
        self.params = nn.Parameter(optim_embeds)

    @torch.no_grad()
    def reinit(self):
        params = self.params[0]
        # 0 grad should have been called before this.
        assert params.grad is None

        self.params.copy_(self.init_params)


Params = Union[HardParams, SoftParams]  # Added for easier typing
ModelBaseT = TypeVar("ModelBaseT", bound="ModelBase")  # For typing load_model


@dataclass
class ForwardReturn:
    """Container for outputs from a model's forward pass.

    Note that we split up model inputs into input and target,
    where input is the user prompt and target is the desired
    output.

    Attributes:
        target_ids: Target token IDs
        target_logits: Logits for target sequence
        target_reps: Hidden representations for target sequence
        input_logits: Logits for input sequence
        input_reps: Hidden representations for input sequence
        loss_mask: Optional mask for loss computation showing padd tokens to ignore (if target_ids is provided)
        loss: Optional loss values (if target_ids is provided)
        input_embeds: Optional input embeddings (for debugging)
        input_ids: Optional input token IDs (for debugging)
        raw_attn_mask: Optional raw attention mask (for debugging)
        raw_logits: Optional raw logits (for debugging)
    """

    target_ids: Int64[Tensor, "b_size target_len"]
    target_logits: Float[Tensor, "b_size target_len vocab_size"]
    target_reps: Float[Tensor, "b_size layers target_len hidden_size"]
    input_logits: Float[Tensor, "b_size input_len vocab_size"]
    input_reps: Float[Tensor, "b_size input_len hidden_size"]
    loss_mask: Optional[Bool[Tensor, "b_size seq_len"]] = None
    loss: Optional[Float[Tensor, "b_size"]] = None
    # These are provided for debugging
    input_embeds: Optional[Float[Tensor, "b_size input_len"]] = None
    input_ids: Optional[Int64[Tensor, "b_size input_len"]] = None
    raw_attn_mask: Optional[Bool[Tensor, "b_size input_len+target_len"]] = None
    raw_logits: Optional[Tensor] = None
    position_ids: Optional[Tensor] = None


@dataclass
class GenReturn:
    """Container for outputs from a model's generation pass.

    Attributes:
        input_text: Original input text
        gen_text: Generated text
        input_ids: Input token IDs
        gen_ids: Generated token IDs
        input_reps: Hidden representations of input
        gen_reps: Hidden representations of generated text
        gen_mask: Mask indicating valid generated tokens that are not padding
    """

    input_text: List[str]
    gen_text: List[str]
    input_ids: Int64[Tensor, "b_size input_len"]
    gen_ids: Int64[Tensor, "b_size gen_len"]
    input_reps: Float[Tensor, "b_size layers input_len hidden_size"]
    gen_reps: Float[Tensor, "b_size layers gen_len hidden_size"]
    gen_mask: Bool[Tensor, "b_size gen_len"]


@dataclass
class ModelConfig:
    """Configuration for model initialization.

    Attributes:
        model_dtype: Data type for model parameters
        prompt_init: Optional initial prompt text
        requires_grad: Whether model parameters require gradients
        device: Device to place model on
    """

    model_dtype: torch.dtype = torch.half
    prompt_init: Optional[str] = None
    requires_grad: bool = False
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ModelBase(ABC, nn.Module):
    """Abstract base class for language models.

    This class defines the interface for language models, including methods for tokenization,
    forward passes, and generation. It supports both discrete token and continuous embedding
    optimization.

    If you want to add a new model, you should inherit from this class and implement the abstract
    methods.
    """

    tokenizer: PreTrainedTokenizer

    @property
    def device(self) -> torch.device:
        """Get the device where model parameters are stored."""
        return next(self.parameters()).device

    @abstractmethod
    def tokenize(
        self,
        text: Union[str, List[str]],
        add_chat_template: bool,
        max_length: Optional[int] = None,
        pad_to_max_length: bool = False,
        add_special_tokens: bool = True,
    ) -> Tuple[Int64[Tensor, "b_size seq_len"], Bool[Tensor, "b_size seq_len"]]:
        """Tokenize input text.

        Args:
            text: Input string or list of strings
            add_chat_template: Whether to add chat template formatting
            max_length: Maximum sequence length
            pad_to_max_length: Whether to pad sequences to max_length
            add_special_tokens: Whether to add special tokens like BOS/EOS

        Returns:
            Tuple of (input_ids, attention_mask)
        """
        ...

    @abstractmethod
    def to_string(
        self,
        input_ids: Int64[Tensor, "b_size seq_len"],
        skip_special_tokens: bool = True,
    ) -> List[str]:
        """Converts input_ids to string."""
        ...

    @abstractmethod
    def forward_from_embeds(
        self,
        input_embeds: Float[Tensor, "b_size seq_len hidden_size"],
        input_attn_mask: Bool[Tensor, "b_size seq_len"],
        target_ids: Optional[Int64[Tensor, "b_size seq_len"]] = None,
        target_attn_mask: Optional[Bool[Tensor, "b_size seq_len"]] = None,
    ) -> ForwardReturn:
        """Forward pass starting from embedding vectors.

        Args:
            input_embeds: Input embeddings
            input_attn_mask: Attention mask for inputs
            target_ids: Optional target token IDs
            target_attn_mask: Optional attention mask for targets (what tokens to ignore)

        Returns:
            ForwardReturn containing model outputs
        """
        ...

    @abstractmethod
    def forward_from_ids(
        self,
        input_ids: Int64[Tensor, "b_size seq_len"],
        input_attn_mask: Bool[Tensor, "b_size seq_len"],
        target_ids: Optional[Int64[Tensor, "b_size seq_len"]] = None,
        target_attn_mask: Optional[Bool[Tensor, "b_size seq_len"]] = None,
        use_tunable_params: bool = True,
    ) -> ForwardReturn:
        """Return per token logits and reps from forward pass on ids.

        Args:
            input_ids: Input token IDs
            input_attn_mask: Attention mask for inputs
            target_ids: Optional target token IDs
            target_attn_mask: Optional attention mask for targets (what tokens to ignore)
            use_tunable_params: Whether to use tunable parameters during the forward pass

        Returns:
            ForwardReturn containing model outputs
        """
        ...

    @abstractmethod
    def forward_from_string(
        self,
        input_text: Union[str, List[str]],
        target_text: Optional[Union[str, List[str]]] = None,
        add_chat_template: bool = True,
        use_tunable_params: bool = True,
    ) -> ForwardReturn:
        """Return per token logits and reps from forward pass on text.

        Args:
            input_text: Input string or list of strings
            target_text: Optional target string or list of strings
            add_chat_template: Whether to add chat template formatting
            use_tunable_params: Whether to use tunable parameters during the forward pass

        Returns:
            ForwardReturn containing model outputs
        """
        ...

    @abstractmethod
    def generate_from_ids(
        self,
        input_ids: Int64[Tensor, "b_size seq_len"],
        input_attn_mask: Bool[Tensor, "b_size seq_len"],
        max_new_tokens: int = 20,
        use_tunable_params: bool = True,
        **generate_kwargs
    ) -> GenReturn:
        """Generate text from input IDs using the model's generate function.

        Args:
            input_ids: Input token IDs
            input_attn_mask: Attention mask for inputs
            max_new_tokens: Maximum number of new tokens to generate
            use_tunable_params: Whether to use tunable parameters during generation
            **generate_kwargs: Additional keyword arguments for the generate function
                (its not pretty but sometimes useful when using hf generate)

        Returns:
            GenReturn containing generated text and other outputs
        """
        ...

    @abstractmethod
    def generate_from_string(
        self,
        input_text: Union[str, List[str]],
        max_new_tokens: int = 20,
        use_tunable_params: bool = True,
        add_chat_template: bool = True,
        **generate_kwargs
    ) -> GenReturn:
        """Generate text from input string(s) using the model's generate function.

        Args:
            input_text: Input string or list of strings
            max_new_tokens: Maximum number of new tokens to generate
            use_tunable_params: Whether to use tunable parameters during generation
            add_chat_template: Whether to add chat template formatting
            **generate_kwargs: Additional keyword arguments for the generate function
                (its not pretty but sometimes useful when using hf generate)
        """
        ...

    @abstractmethod
    def init_tunable_params(self) -> Params:
        """Initialize tunable parameters for optimization and return them.

        Returns:
            New parameter object (HardParams or SoftParams)
        """
        ...

    @classmethod
    @abstractmethod
    def load_model(
        cls: Type[ModelBaseT],
        path: Path,
        config: ModelConfig = ModelConfig(),
    ) -> ModelBaseT:
        """Load model from disk.

        Args:
            path: Path to saved model
            config: Model configuration

        Returns:
            Loaded model instance
        """
        ...
