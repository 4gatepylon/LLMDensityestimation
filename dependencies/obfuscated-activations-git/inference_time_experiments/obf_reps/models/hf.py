from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Type, TypeVar, Union

import torch
from jaxtyping import Bool, Float, Int64
from torch import Tensor, embedding, nn
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer

from obf_reps.models import (
    ForwardReturn,
    GenReturn,
    HardParams,
    ModelBase,
    ModelConfig,
    SoftParams,
)

HFModelBaseT = TypeVar("HFModelBaseT", bound="HFModelBase")


class HFModelBase(ModelBase, ABC):
    """Base class for Hugging Face language models.

    This class implements the ModelBase interface for Hugging Face models,
    providing common functionality for tokenization, forward passes, and generation.
    When implimenting a new hugginface model, inherit from this class.

    Args:
        model: Hugging Face model instance
        tokenizer: Hugging Face tokenizer
        config: Model configuration
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: PreTrainedTokenizer,
        config: ModelConfig,
    ):

        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.dtype = config.model_dtype
        self.prompt_init = config.prompt_init

        self.tunable_params = self.init_tunable_params()
        self.config = config

    def tokenize(
        self,
        text: Union[str, List[str]],
        add_chat_template: bool,
        max_length: Optional[int] = None,
        pad_to_max_length: bool = False,
        add_special_tokens: bool = True,
    ) -> Tuple[Int64[Tensor, "b_size seq_len"], Bool[Tensor, "b_size seq_len"]]:
        """Tokenizes text into input_ids and attention_mask.

        Args:
            text: string to tokenize.
            add_chat_template: whether to treat text as a user input and wrap
                it in the required templating e.g. "<bos>User: {text}<eot>".
                This superceeds add_special_tokens, that is, special tokens
                will always be added.
            max_length: maximum length of tokenization.
            pad_to_max_length: whether to pad to max length.
            add_special_tokens: whether to add special tokens.

        Returns:
            Tuple of input_ids and attention_mask
        """

        if pad_to_max_length:
            assert max_length is not None, "max_length must be set when using pad_to_max_lenght"

        if isinstance(text, str):
            text = [text]

        if add_chat_template:
            # Assume that text argument is the user input
            batched_messages = [[{"role": "user", "content": msg}] for msg in text]
            text: List[str] = self.tokenizer.apply_chat_template(
                batched_messages,
                tokenize=False,
                add_generation_prompt=True,  # type: ignore
            )

            # Override addition of special tokens, as this has already been done
            add_special_tokens = False

        inputs = self.tokenizer(
            text=text,
            max_length=max_length,
            padding="max_length" if pad_to_max_length else "longest",
            truncation=True,
            return_tensors="pt",
            add_special_tokens=add_special_tokens,
        )

        return inputs["input_ids"].to(self.device), inputs["attention_mask"].to(self.device)

    def to_string(
        self,
        input_ids: Int64[Tensor, "b_size seq_len"],
        skip_special_tokens: bool = True,
    ) -> List[str]:
        """Converts input_ids to string."""

        return self.tokenizer.batch_decode(
            input_ids,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=True,
        )

    def forward_from_string(
        self,
        input_text: Union[str, List[str]],
        target_text: Optional[Union[str, List[str]]] = None,
        add_chat_template: bool = True,
        use_tunable_params: bool = True,
        debug: bool = False,
    ) -> ForwardReturn:
        """Return per token logits and reps from forward pass on text.

        Args:
            input_text: input text. If using tunable prompt, this will
                be added after the input text.
            target_text: target that the model should predict after
                input_text (+ tunable prompt). If provided, then
                a loss mask over tokens and loss will be returned.
            add_chat_template: whether to treat input_text as a user input and wrap
                it in the required templating e.g. "<bos>User: {text}<eot>".
                This superceeds add_special_tokens, that is, special tokens
                will always be added.
            use_tunable_params: whether to use tunable parameters during the forward pass
            debug: whether to return additional debugging information
        """

        if isinstance(input_text, str):
            input_text = [input_text]
        if isinstance(target_text, str):
            target_text = [target_text]

        input_ids, input_attn_mask = self.tokenize(input_text, add_chat_template=add_chat_template)

        target_ids = None
        target_attn_mask = None
        if target_text is not None and target_text != "":
            target_ids, target_attn_mask = self.tokenize(
                target_text,
                add_special_tokens=False,
                add_chat_template=False,
                pad_right=True,
            )

        return self.forward_from_ids(
            input_ids=input_ids,
            input_attn_mask=input_attn_mask,
            target_ids=target_ids,
            target_attn_mask=target_attn_mask,
            use_tunable_params=use_tunable_params,
        )

    # ----- Load Model -----#
    @classmethod
    def load_model(
        cls: Type[HFModelBaseT],
        path: Path,
        config: ModelConfig = ModelConfig(),
    ) -> HFModelBaseT:
        """Load model
        Args:
            path: path to saved model
            model_dtype: dtype of model
            requires_grad: whether to compute gradients for model params
        """

        model = AutoModelForCausalLM.from_pretrained(
            path, device_map=config.device, torch_dtype=config.model_dtype
        )
        model.eval()

        if not config.requires_grad:
            for p in model.parameters():
                p.requires_grad = False

        tokenizer = AutoTokenizer.from_pretrained(path)

        # Pad from left as we are adding suffixes
        tokenizer.padding_side = "left"
        if tokenizer.pad_token:
            pass
        elif tokenizer.unk_token:
            tokenizer.pad_token_id = tokenizer.unk_token_id
        elif tokenizer.eos_token:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

        model.generation_config.pad_token_id = tokenizer.pad_token_id

        return cls(model, tokenizer, config)


class HFModelPrompted(HFModelBase, ABC):
    """HF model where tunable parameters are added to the input.

    This class extends HFModelBase to support insertion of tunable parameters
    (either discrete tokens or continuous embeddings) at specified locations
    in the input sequence marked by OPT_LOC_TOKEN. For now, this
    is hardcoded to be AFTER the input text, i.e. it is added as
    a suffix to the end of the user query.

    Args:
        model: Hugging Face model instance
        tokenizer: Hugging Face tokenizer
        config: Model configuration
    """

    OPT_LOC_TOKEN = "<|optim-location|>"  # Note in __init__ this is added to the tokenizer

    def __init__(
        self,
        model: nn.Module,
        tokenizer: PreTrainedTokenizer,
        config: ModelConfig,
    ):
        super().__init__(model, tokenizer, config)
        self.tokenizer.add_special_tokens({"additional_special_tokens": [self.OPT_LOC_TOKEN]})
        self.opt_loc_token_id = self.tokenizer.convert_tokens_to_ids(self.OPT_LOC_TOKEN)

    def tokenize(
        self,
        text: Union[str, List[str]],
        add_chat_template: bool,
        max_length: Optional[int] = None,
        pad_to_max_length: bool = False,
        add_special_tokens: bool = True,
        pad_right: bool = False,
    ) -> Tuple[Int64[Tensor, "b_size seq_len"], Bool[Tensor, "b_size seq_len"]]:
        """Tokenizes text into input_ids and attention_mask.

        Args:
            text: string to tokenize.
            add_chat_template: whether to treat text as a user input and wrap
                it in the required templating e.g. "<bos>User: {text}<eot>".
                This superceeds add_special_tokens, that is, special tokens
                will always be added.
            max_length: maximum length of tokenization.
            pad_to_max_length: whether to pad to max length.
            add_special_tokens: whether to add special tokens.
            pad_right: whether to pad on the right side instead of left.
                We default to padding on the left as it is better for generation.

        Returns:
            Tuple of input_ids and attention_mask
        """

        if isinstance(text, str):
            text = [text]

        if add_chat_template:
            # Add in the optimization location if adding chat template
            text = [x + self.OPT_LOC_TOKEN for x in text]

        # We default to padding on the left, as it is better for generation.
        if pad_right:
            self.tokenizer.padding_side = "right"

        output = super().tokenize(
            text,
            add_chat_template,
            max_length,
            pad_to_max_length,
            add_special_tokens,
        )
        # reset tokenizer side
        self.tokenizer.padding_side = "left"

        return output

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

        attention_mask = input_attn_mask

        if (
            target_ids is not None
            and target_attn_mask is not None
            and target_ids.numel() > 0
            and target_attn_mask.numel() > 0
        ):
            # Prepare input_text + tunable prompt + target_text

            target_embeds: Float[Tensor, "b_size target_seq_len hidden_dim"] = (
                self.model.get_input_embeddings()(target_ids)
            )

            # Calculate the loss mask
            batch_size = input_embeds.shape[0]
            input_seq_len = input_embeds.shape[1]
            input_loss_mask = (
                torch.ones(batch_size, input_seq_len, dtype=torch.bool, device=self.device) * -100
            )
            target_loss_mask = target_ids.clone()
            target_loss_mask[target_attn_mask == 0] = -100

            hf_labels = torch.cat([input_loss_mask, target_loss_mask], dim=1)

            # Combine input_embeds and target_embeds
            input_embeds = torch.cat([input_embeds, target_embeds], dim=1)

            # Combine attention_masks
            attention_mask = torch.cat([attention_mask, target_attn_mask], dim=1)

            # Get position ids
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, -1)
            position_ids = position_ids.contiguous()

            # Collate the model return
            raw_output = self.model(
                inputs_embeds=input_embeds,
                labels=hf_labels,
                attention_mask=attention_mask,
                position_ids=position_ids,
                output_hidden_states=True,
            )

            hf_loss = raw_output.loss
            target_len = target_ids.shape[1]
            logits: Float[Tensor, "b_size seq_len vocab_size"] = raw_output.logits
            prediction_logits = logits[:, -target_len - 1 : -1, :]
            input_logits = logits[:, : -target_len - 1, :]

            reps: Float[Tensor, "b_size layers seq_len hidden_size"] = torch.stack(
                raw_output.hidden_states, dim=1
            )

            prediction_reps = reps[:, :, -target_len:, :]
            input_reps = reps[:, :, :-target_len, :]

            loss_mask = hf_labels[:, -target_len:]
            loss_mask[loss_mask != -100] = 1
            loss_mask[loss_mask == -100] = 0
            loss_mask = loss_mask.bool()

            output = ForwardReturn(
                target_ids=target_ids,
                target_logits=prediction_logits,
                target_reps=prediction_reps,
                input_logits=input_logits,
                input_reps=input_reps,
                loss_mask=loss_mask,
                loss=hf_loss,
            )

        else:
            raw_output = self.model(
                inputs_embeds=input_embeds,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )

            # Use torch.stack as we want to create a new dimension for layers
            reps: Float[Tensor, "b_size layers seq_len hidden_size"] = torch.stack(
                raw_output.hidden_states, dim=1
            )
            prediction_reps = reps[:, :, -1:, :]
            input_reps = reps

            logits = raw_output.logits
            prediction_logits = logits[:, -1:, :]
            input_logits = logits[:, :-1, :]

            output = ForwardReturn(
                target_ids=target_ids,
                target_logits=prediction_logits,
                target_reps=prediction_reps,
                input_logits=input_logits,
                input_reps=input_reps,
            )

        return output

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

        input_embeds, input_attn_mask = self._convert_ids_to_input_embeds(
            input_ids, input_attn_mask, use_tunable_params
        )

        output = self.forward_from_embeds(
            input_embeds=input_embeds,
            input_attn_mask=input_attn_mask,
            target_ids=target_ids,
            target_attn_mask=target_attn_mask,
        )

        return output

    def generate_from_ids(
        self,
        input_ids: Int64[Tensor, "b_size input_seq_len"],
        input_attn_mask: Bool[Tensor, "b_size input_seq_len"],
        max_new_tokens: int = 20,
        use_tunable_params: bool = True,
        **generate_kwargs,
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

        input_embeds, attention_mask = self._convert_ids_to_input_embeds(
            input_ids, input_attn_mask, use_tunable_params
        )

        # Generate output using the model's generate function
        output = self.model.generate(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            output_hidden_states=True,
            return_dict_in_generate=True,
            min_new_tokens=5,
            **generate_kwargs,
        )

        batch_size: int = input_ids.shape[0]
        input_seq_len: int = input_ids.shape[1]

        # When forward with input_embeds, model should only return ids for generation
        gen_ids: Int64[Tensor, "b_size gen_len"] = output.sequences

        # Get text and reps
        gen_text: List[str] = self.tokenizer.batch_decode(gen_ids, skip_special_tokens=False)
        input_text: List[str] = self.tokenizer.batch_decode(input_ids, skip_special_tokens=False)

        hidden_states: List[List[Tensor]] = output.hidden_states
        assert isinstance(
            hidden_states, (list, tuple)
        ), "Hidden states are not in the correct datastructure"
        assert isinstance(
            hidden_states[0], (list, tuple)
        ), "Hidden states are not in the correct datastructure"
        assert isinstance(
            hidden_states[0][0], torch.Tensor
        ), "Hidden states are not in the correct datastructure"

        input_reps_list: List[Float[Tensor, "b_size input_seq_len hidden_size"]] = hidden_states[0]
        input_reps: Float[Tensor, "b_size layers input_seq_len hidden_size"] = torch.stack(
            input_reps_list, dim=1
        )  # Create new layer dimension
        assert (
            input_reps.shape[0] == batch_size
        ), "Forming input_reps failed. Model likely returned hidden states in unexpected format."

        gen_reps_list: List[Float[Tensor, "b_size layers hiden_size"]] = [
            torch.cat(x, dim=1) for x in hidden_states[1:]
        ]  # Note formatted different to above
        gen_reps: Float[Tensor, "b_size layers gen_seq_len hidden_size"] = torch.stack(
            gen_reps_list, dim=2
        )

        assert (
            gen_reps.shape[0] == batch_size
        ), "Forming gen_reps failed. Model likely returned hidden states in unexpected format."
        # Note that if you generate n tokens, you will have n-1 reps as no reps for the final token that was just generated.
        assert (
            gen_reps.shape[1] == input_reps.shape[1]
        ), "Number of layers in gen_reps and input_reps does not match."

        pad_token_id = self.tokenizer.pad_token_id
        eot_token_id = self.tokenizer.eos_token_id

        gen_mask = (gen_ids != pad_token_id) & (gen_ids != eot_token_id)

        return GenReturn(
            input_text=input_text,
            gen_text=gen_text,
            input_ids=input_ids,
            gen_ids=gen_ids,
            input_reps=input_reps,
            gen_reps=gen_reps,
            gen_mask=gen_mask,
        )

    def generate_from_string(
        self,
        input_text: Union[str, List[str]],
        max_new_tokens: int = 20,
        use_tunable_params: bool = True,
        add_chat_template: bool = True,
        **generate_kwargs,
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

        # Tokenize input text
        input_ids, input_attn_mask = self.tokenize(
            input_text,
            add_chat_template=add_chat_template,
        )

        generated_texts = self.generate_from_ids(
            input_ids=input_ids,
            input_attn_mask=input_attn_mask,
            max_new_tokens=max_new_tokens,
            use_tunable_params=use_tunable_params,
            **generate_kwargs,
        )

        return generated_texts

    def _convert_ids_to_input_embeds(
        self,
        input_ids: Int64[Tensor, "b_size seq_len"],
        input_attn_mask: Bool[Tensor, "b_size seq_len"],
        use_tunable_params: bool,
    ) -> Tuple[
        Float[Tensor, "b_size new_seq_len hidden_dim"],
        Bool[Tensor, "b_size new_seq_len"],
    ]:
        """Convert input token IDs to embeddings, optionally inserting tunable parameters.

        This method handles the conversion of input token IDs to embeddings while managing
        the insertion of tunable parameters (if enabled) at locations marked by opt_loc_token_id.
        The tunable parameters are inserted at these marked positions, replacing the marker tokens.

        Args:
            input_ids: Input token IDs with optional opt_loc_token_id markers
            input_attn_mask: Attention mask corresponding to input_ids
            use_tunable_params: Whether to insert tunable parameters at marked locations

        Returns:
            Tuple containing:
                - Embedding vectors for the input sequence with any tunable parameters inserted
                - Updated attention mask accounting for inserted parameters

        Raises:
            ValueError: If use_tunable_params is True but no opt_loc_token_id markers are found
        """

        if not torch.any(input_ids == self.opt_loc_token_id):
            # No locations to insert tunable params
            if use_tunable_params:
                raise ValueError(
                    "use_tunable_params set to true but not locations in input_ids for them to go."
                )
            else:
                # Good to just convert to input_embeds
                input_embeds = self.model.get_input_embeddings()(input_ids)
                attention_mask = input_attn_mask
        else:  # Tunable params are located in the input
            # Assert that each row of input_ids has a self.opt_loc_token_id token
            # and that they are all in the same place
            locations = (input_ids == self.opt_loc_token_id).nonzero(as_tuple=False)
            unique_columns = torch.unique(locations[:, 1])
            counts = torch.sum(input_ids == self.opt_loc_token_id, dim=1)
            assert torch.all(
                counts == 1
            ), "Not exactly one instance of self.opt_loc_token_id in each row."
            assert (
                len(unique_columns) == 1
            ), "self.opt_loc_token_id is not in the same column in all rows."

            insertion_column = unique_columns.item()
            left_input_ids = input_ids[:, :insertion_column]  # All columns before the N column
            right_input_ids = input_ids[
                :, insertion_column + 1 :
            ]  # All columns after the N column
            left_attention_mask = input_attn_mask[:, :insertion_column]
            right_attention_mask = input_attn_mask[:, insertion_column + 1 :]

            assert not torch.any(
                left_input_ids == self.opt_loc_token_id
            ), "Failed to remove opt_loc_token_id"
            assert not torch.any(
                right_input_ids == self.opt_loc_token_id
            ), "Failed to remove opt_loc_token_id"

            left_input_embeds: Float[Tensor, "b_size l_seq_len hidden_dim"] = (
                self.model.get_input_embeddings()(left_input_ids)
            )
            right_input_embeds: Float[Tensor, "b_size r_seq_len hidden_dim"] = (
                self.model.get_input_embeddings()(right_input_ids)
            )

            if use_tunable_params:
                # Add in tunable params
                batch_size = input_ids.shape[0]

                param_embeds: Float[Tensor, "1 seq_len hidden_size"] = (
                    self._get_tunable_param_embeds()
                )
                batched_param_embeds: Float[Tensor, "b seq_len hidden_size"] = param_embeds.expand(
                    batch_size, -1, -1
                )

                input_embeds = torch.cat(
                    [left_input_embeds, batched_param_embeds, right_input_embeds], dim=1
                )

                num_params = param_embeds.shape[1]
                soft_toks_am = torch.ones(
                    batch_size,
                    num_params,
                    dtype=input_attn_mask.dtype,
                    device=self.device,
                )
                attention_mask = torch.cat(
                    [left_attention_mask, soft_toks_am, right_attention_mask],
                    dim=1,
                )
            else:
                input_embeds = torch.cat([left_input_embeds, right_input_embeds], dim=1)
                attention_mask = torch.cat([left_attention_mask, right_attention_mask], dim=1)

        return input_embeds, attention_mask

    @abstractmethod
    def _get_tunable_param_embeds(self) -> Float[Tensor, "1 seq_len hidden_size"]:
        """Get the tunable prompt parameters as embeddings."""
        ...


class HFHardPrompted(HFModelPrompted):
    """HF model with discrete token parameters.

    This class implements HFModelPrompted using hard (discrete) token parameters that are optimized
    during training. The parameters are stored as one-hot vectors over the vocabulary.
    """

    def init_tunable_params(self) -> HardParams:
        """Initialize discrete token parameters.

        Returns:
            HardParams object containing one-hot encoded token parameters
        """

        if self.prompt_init is not None:
            prompt_ids, _ = self.tokenize(
                self.prompt_init, add_special_tokens=False, add_chat_template=False
            )
        else:
            self.prompt_init = "x" * 20
            prompt_ids, _ = self.tokenize(
                self.prompt_init, add_special_tokens=False, add_chat_template=False
            )

        return HardParams(prompt_ids, self.model.get_input_embeddings())

    def _get_tunable_param_embeds(self) -> Float[Tensor, "1 seq_len hidden_size"]:
        """Get the tunable prompt parameters as embeddings.

        Returns:
            Tensor of shape (1, seq_len, hidden_size) containing embeddings
        """

        embedding_layer = self.model.get_input_embeddings()
        optim_embeds = self.tunable_params.params @ embedding_layer.weight

        return optim_embeds


class HFSoftPrompted(HFModelPrompted):
    """HF model with continuous embedding parameters.

    This class implements HFModelPrompted using soft (continuous) embedding parameters that are
    optimized during training. The parameters are stored as dense vectors in the embedding space.
    """

    def init_tunable_params(self) -> SoftParams:
        """Initialize continuous embedding parameters.

        Returns:
            SoftParams object containing embedding parameters
        """

        embedding_layer = self.model.get_input_embeddings()
        if self.prompt_init is not None:
            prompt_ids, _ = self.tokenize(
                self.prompt_init, add_special_tokens=False, add_chat_template=False
            )
        else:
            prompt_ids = torch.randint(
                0, embedding_layer.num_embeddings, (1, 100), device=self.device
            )

        return SoftParams(prompt_ids, embedding_layer)

    def _get_tunable_param_embeds(self) -> Float[Tensor, "1 seq_len hidden_size"]:
        """Get the tunable prompt parameters as embeddings.

        Returns:
            Tensor of shape (1, seq_len, hidden_size) containing embeddings
        """
        return self.tunable_params.params  # type: ignore
