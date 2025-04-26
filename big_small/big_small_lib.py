from __future__ import annotations
from pathlib import Path
import uuid
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GPT2Tokenizer,
    GPT2TokenizerFast,
)
import torch
import torch.nn as nn
import einops
from typing import Tuple, List, Optional
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint
from jaxtyping import Float, Int
import matplotlib.pyplot as plt
import numpy as np


class Utility:
    @staticmethod
    def get_hooked_activations(
        tokens: Int[torch.Tensor, "batch pos"],
        model: HookedTransformer | AutoModelForCausalLM,
        hook_name: str,
        use_post_hook: bool = False,
    ) -> Float[torch.Tensor, "batch pos d_model"]:
        if isinstance(model, HookedTransformer):
            # NOTE: pre-hook vs. post-hook is ignored
            return Utility.get_hooked_activations_from_tl(tokens, model, hook_name)
        else:
            return Utility.get_hooked_activations_from_hf(
                tokens, model, hook_name, use_post_hook
            )

    @staticmethod
    def get_hooked_activations_from_tl(
        tokens: Int[torch.Tensor, "batch pos"], model: HookedTransformer, hook_name: str
    ) -> Float[torch.Tensor, "batch pos d_model"]:
        assert isinstance(tokens, torch.Tensor)
        assert tokens.ndim == 2
        assert isinstance(model, HookedTransformer)
        assert isinstance(hook_name, str)
        with torch.no_grad():
            outs = []
            def collect_hook(act, hook: HookPoint):
                outs.append(act.detach().clone())
                return act
            model.run_with_hooks(
                tokens,
                return_type=None,
                fwd_hooks=[(hook_name, collect_hook)]
            )
            assert len(outs) == 1
            return outs[0]

    @staticmethod
    def get_hooked_activations_from_hf(
        tokens: Int[torch.Tensor, "batch pos"],
        model: AutoModelForCausalLM | nn.Module,
        hook_name: str,
        use_post_hook: bool = False,
    ) -> Float[torch.Tensor, "batch pos d_model"]:
        assert isinstance(tokens, torch.Tensor)
        assert tokens.ndim == 2
        assert isinstance(hook_name, str)
        with torch.no_grad():
            ################ SET UP THE HOOKS #################
            outs = []
            def forward_hook(module, input, output):
                nonlocal outs
                if not isinstance(output, torch.Tensor):
                    output = output[0]
                assert isinstance(output, torch.Tensor)
                outs.append(output.detach().clone())
            def forward_pre_hook(module, input, *args):
                nonlocal outs
                if not isinstance(input, torch.Tensor):
                    input = input[0]
                assert isinstance(input, torch.Tensor)
                outs.append(input.detach().clone())
            ################ GET THE ACTIVATIONS #################
            registered: bool = False
            for name, module in model.named_modules():
                if name == hook_name:
                    if use_post_hook:
                        handle = module.register_forward_hook(forward_hook)
                    else:
                        handle = module.register_forward_pre_hook(forward_pre_hook)
                    registered = True
                    break
            if not registered:
                raise ValueError(f"Hook name {hook_name} not found in model\n{'\n'.join(map(lambda x: str(x[0]), model.named_modules()))}") # fmt: skip
            try:
                model(tokens)
            finally:
                handle.remove()
            assert len(outs) == 1
            return outs[0]


class BigSmallLitmusTester:
    """
    A class to basically generate a bunch of sentences with big and small words.

    This is written in a monolithic AI aided way so it's not really good code :P
    But it should work OK!

    NOTE it does not use batching cuz... YOLO (but in theory you could improve this
    without the issue of batching by using https://huggingface.co/blog/sirluk/llm-sequence-packing)
    """

    # Top 10 most commonly used "big" words
    # According to Claude
    big_words = [
        "big",
        "large",
        "huge",
        "massive",
        "enormous",
        "giant",
        "great",
        "vast",
        "wide",
        "substantial",
    ]

    # Top 10 most commonly used "small" words
    # According to Claude
    small_words = [
        "small",
        "little",
        "tiny",
        "mini",
        "slight",
        "petite",
        "minor",
        "compact",
        "slim",
        "modest",
    ]
    # slotin keyword is what we do .replace for
    # (can be anything but needs to be different and
    # appear once)
    slotin_keyword = "1E427FD5-9D52-479C-ADBD-0A3D23B3BB5E"

    # Some good templates to basically slot in
    # big or small words to generate realistic stentences
    # According to Claude
    sentence_templates = [
        "I need a {} box to store my collection.".format(slotin_keyword),  # fmt: skip
        "The {} elephant at the zoo attracted everyone's attention.".format(
            slotin_keyword
        ),  # fmt: skip
        "She lived in a {} apartment in the city center.".format(
            slotin_keyword
        ),  # fmt: skip
        "That's a {} problem compared to what we faced last year.".format(
            slotin_keyword
        ),  # fmt: skip
        "I'm looking for a {} car that's easy to park.".format(
            slotin_keyword
        ),  # fmt: skip
        "The company made a {} profit this quarter.".format(
            slotin_keyword
        ),  # fmt: skip
        "He had a {} smile on his face when he heard the news.".format(
            slotin_keyword
        ),  # fmt: skip
        "We need a {} table for the dining room.".format(slotin_keyword),  # fmt: skip
        "The chef prepared a {} portion of pasta.".format(slotin_keyword),  # fmt: skip
        "There was a {} gap between the fence posts.".format(slotin_keyword),
    ]

    def __init__(
        self, tokenizer: GPT2Tokenizer | GPT2TokenizerFast | str, debug: bool = False
    ):  # , model: str, temperature: float, max_tokens: int) -> None:
        assert all(
            template.count(self.slotin_keyword) == 1
            for template in self.sentence_templates
        )
        self.debug = debug
        # NOTE: we rstrip because we will be taking the tokens with the space before them
        # (ugh gpt)
        if self.debug:
            print("=" * 50 + " Generating Strings Properly " + "=" * 50)
        self.prefixes = [
            template.split(self.slotin_keyword, 1)[0].rstrip()
            for template in self.sentence_templates
        ]
        self.suffixes = [
            template.split(self.slotin_keyword, 1)[1]
            for template in self.sentence_templates
        ]
        self.big_sentences_str = []
        self.small_sentences_str = []
        # NOTE: size choice is outer, then template then word
        for size_words, size_sentences_str in [
            (self.big_words, self.big_sentences_str),
            (self.small_words, self.small_sentences_str),
        ]:
            for template in self.sentence_templates:
                for word in size_words:
                    size_sentences_str.append(
                        template.replace(self.slotin_keyword, word)
                    )

        assert len(self.big_sentences_str) == len(self.big_words) * len(
            self.sentence_templates
        )
        assert len(self.small_sentences_str) == len(self.small_words) * len(
            self.sentence_templates
        )
        assert len(self.big_sentences_str) == len(
            self.small_sentences_str
        )  # less important

        if self.debug:
            print("=" * 50 + " Sanity Checking strings " + "=" * 50)  # DEBUG
            print(self.big_sentences_str[0])  # DEBUG
            print(self.small_sentences_str[0])  # DEBUG
            print("|" + self.prefixes[0] + "|")  # DEBUG
            print("|" + self.suffixes[0] + "|")  # DEBUG

        self.tokenizer = (
            tokenizer
            if isinstance(tokenizer, AutoTokenizer)
            else AutoTokenizer.from_pretrained(tokenizer)
        )
        if not isinstance(self.tokenizer, GPT2TokenizerFast) and not isinstance(
            self.tokenizer, GPT2Tokenizer
        ):
            raise NotImplementedError("Only gpt2 supported")

        # Create tokenized versions of the prefixes and suffixes
        if self.debug:
            print("=" * 50 + " Tokenizing prefixes and suffixes " + "=" * 50)
        self.prefixes_tokenized = [
            self.tokenizer.encode(prefix) for prefix in self.prefixes
        ]
        self.suffixes_tokenized = [
            self.tokenizer.encode(suffix) for suffix in self.suffixes
        ]

        if self.debug:
            print("=" * 50 + " Tokenizing big and small words " + "=" * 50)
        # TODO(Adriano) do we actually care about adding spaces before or not? should we care about token indices? idk
        self.big_words_tokenized = [
            self.tokenizer.encode(" " + big_word.strip()) for big_word in self.big_words
        ]
        self.small_words_tokenized = [
            self.tokenizer.encode(" " + small_word.strip())
            for small_word in self.small_words
        ]

        if self.debug:
            print("=" * 50 + " Generating the sentences (tokenized) " + "=" * 50)
        # Store big sentences and small sentences tokenized
        self.big_sentences_tokenized = []
        self.small_sentences_tokenized = []
        # Store big  and small words start and end token indices
        self.big_words_start_token_indices = []
        self.big_words_end_token_indices = []
        self.small_words_start_token_indices = []
        self.small_words_end_token_indices = []

        # Create this
        assert len(self.prefixes_tokenized) == len(self.suffixes_tokenized)
        # NOTE: once again size choice is outer, then template then word
        for (
            size_words,
            size_sentences,
            size_words_start_token_indices,
            size_words_end_token_indices,
        ) in [
            (
                self.big_words_tokenized,
                self.big_sentences_tokenized,
                self.big_words_start_token_indices,
                self.big_words_end_token_indices,
            ),
            (
                self.small_words_tokenized,
                self.small_sentences_tokenized,
                self.small_words_start_token_indices,
                self.small_words_end_token_indices,
            ),
        ]:
            for prefix, suffix in zip(self.prefixes_tokenized, self.suffixes_tokenized):
                for word in size_words:
                    sentence = prefix + word + suffix
                    size_sentences.append(sentence)
                    size_words_start_token_indices.append(len(prefix))
                    size_words_end_token_indices.append(len(prefix) + len(word))
        assert (
            len(self.big_sentences_tokenized)
            == len(self.big_words_start_token_indices)
            == len(self.big_words_end_token_indices)
        )
        assert (
            len(self.small_sentences_tokenized)
            == len(self.small_words_start_token_indices)
            == len(self.small_words_end_token_indices)
        )
        assert len(self.big_sentences_tokenized) == len(self.big_words) * len(
            self.sentence_templates
        )
        assert len(self.small_sentences_tokenized) == len(self.small_words) * len(
            self.sentence_templates
        )
        assert len(self.big_sentences_tokenized) == len(
            self.small_sentences_tokenized
        )  # less important

        if self.debug:
            print("=" * 50 + " Sanity Checking enc/dec " + "=" * 50)  # DEBUG
        self.big_sentences_detokenized = [
            self.tokenizer.decode(sentence) for sentence in self.big_sentences_tokenized
        ]
        self.small_sentences_detokenized = [
            self.tokenizer.decode(sentence)
            for sentence in self.small_sentences_tokenized
        ]
        if self.debug:
            print(self.big_sentences_detokenized[0])  # DEBUG
            print(self.small_sentences_detokenized[0])  # DEBUG
        assert all(
            self.big_sentences_detokenized[i] == self.big_sentences_str[i]
            for i in range(len(self.big_sentences_detokenized))
        )
        assert all(
            self.small_sentences_detokenized[i] == self.small_sentences_str[i]
            for i in range(len(self.small_sentences_detokenized))
        )

        # Make sure everything is a tensor
        if not all(
            isinstance(tensor, torch.Tensor) for tensor in self.big_sentences_tokenized
        ):
            assert all(
                isinstance(l, list) and all(isinstance(i, int) for i in l)
                for l in self.big_sentences_tokenized
            )
            self.big_sentences_tokenized = [
                torch.tensor(l) for l in self.big_sentences_tokenized
            ]
        if not all(
            isinstance(tensor, torch.Tensor)
            for tensor in self.small_sentences_tokenized
        ):
            assert all(
                isinstance(l, list) and all(isinstance(i, int) for i in l)
                for l in self.small_sentences_tokenized
            )
            self.small_sentences_tokenized = [
                torch.tensor(l) for l in self.small_sentences_tokenized
            ]
        # Should NOT be batched
        assert all(x.ndim == 1 for x in self.big_sentences_tokenized)
        assert all(x.ndim == 1 for x in self.small_sentences_tokenized)

        # Make sure everything is a batched tensor by exploiting the end index
        max_big_end_idx = max(self.big_words_end_token_indices)
        max_small_end_idx = max(self.small_words_end_token_indices)
        assert all(x.shape[0] >= max_big_end_idx for x in self.big_sentences_tokenized) # fmt: skip
        assert all(x.shape[0] >= max_small_end_idx for x in self.small_sentences_tokenized) # fmt: skip

        self.small_sentences_tokenized_pt = torch.stack([s[:max_small_end_idx] for s in self.small_sentences_tokenized], dim=0) # fmt: skip
        self.big_sentences_tokenized_pt = torch.stack([s[:max_big_end_idx] for s in self.big_sentences_tokenized], dim=0) # fmt: skip
        assert (
            self.small_sentences_tokenized_pt.ndim
            == self.big_sentences_tokenized_pt.ndim
            == 2
        )
        assert self.small_sentences_tokenized_pt.shape[0] == len(
            self.small_sentences_str
        )
        assert self.big_sentences_tokenized_pt.shape[0] == len(self.big_sentences_str)
        if self.debug:
            print("=" * 50 + " Shapes of our PTs " + "=" * 50)  # DEBUG
            print(self.small_sentences_tokenized_pt.shape)  # DEBUG
            print(self.big_sentences_tokenized_pt.shape)  # DEBUG

    def get_2d_projections(
        self,
        model: torch.nn.Module,
        # Use nn.Module for affine transform
        projection: torch.Tensor | nn.Module,
        hook_name: str,
        use_post_hook: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the 2D projections of the matrix using the model. The model
        must be a HookedTransformer or it can be an nn.Module (in which case
        we use pt hooks).

        Return tensor that is always n_points x 2 by just batching everything.

        Returns only big small projections.
        """
        assert not isinstance(projection, torch.Tensor) or projection.ndim == 2
        if isinstance(projection, torch.Tensor):
            if projection.shape[0] != 2:
                raise NotImplementedError("Only 2D projections are supported")

        #### GET THE ACTIVATIONS ####
        # 1. Get the activations
        try:
            model_device = model.device
        except:
            model_device = next(model.parameters()).device
        big_tokens_pt = self.big_sentences_tokenized_pt.to(model_device)
        small_tokens_pt = self.small_sentences_tokenized_pt.to(model_device)
        activations_big = Utility.get_hooked_activations(big_tokens_pt, model, hook_name, use_post_hook) # fmt: skip
        activations_small = Utility.get_hooked_activations(small_tokens_pt, model, hook_name, use_post_hook) # fmt: skip
        # 2. Get the RELEVANT activations
        if self.debug:
            print("=" * 50 + " Shapes of our activations big and small from utility " + "=" * 50)  # DEBUG
            print(activations_big.shape)
            print(activations_small.shape)
            print("=" * 100)  # DEBUG
        assert activations_big.ndim == activations_small.ndim == 3  # batch seq d_model
        batch_big, seq_big, d_model = activations_big.shape
        batch_small, seq_small, _ = activations_small.shape
        assert batch_big == len(self.big_words_start_token_indices) == len(self.big_words_end_token_indices) # fmt: skip
        assert batch_small == len(self.small_words_start_token_indices) == len(self.small_words_end_token_indices) # fmt: skip
        assert d_model == _  # same d_model
        # Select the proper seq elements (tokens that are the word indices)
        activations_big_flat = [
            act[start_token_idx:end_token_idx, :].reshape(-1, d_model)
            for act, start_token_idx, end_token_idx in zip(
                activations_big,
                self.big_words_start_token_indices,
                self.big_words_end_token_indices,
            )
        ]
        activations_small_flat = [
            act[start_token_idx:end_token_idx, :].reshape(-1, d_model)
            for act, start_token_idx, end_token_idx in zip(
                activations_small,
                self.small_words_start_token_indices,
                self.small_words_end_token_indices,
            )
        ]
        if self.debug:
            print("=" * 50 + " Shapes of our activations_big_flat (filtered, no cat) lists " + "=" * 50) # DEBUG; fmt: skip
            print([act.shape for act in activations_big_flat])  # DEBUG
            print([act.shape for act in activations_small_flat])  # DEBUG
            print("=" * 100)  # DEBUG
        activations_big_flat_pt = torch.cat(activations_big_flat, dim=0)
        activations_small_flat_pt = torch.cat(activations_small_flat, dim=0)
        assert activations_big_flat_pt.shape[0] >= batch_big  # at least one tok
        assert activations_small_flat_pt.shape[0] >= batch_small  # at least one tok
        assert activations_big_flat_pt.ndim == 2
        assert activations_small_flat_pt.ndim == 2
        assert activations_big_flat_pt.shape[-1] == activations_small_flat_pt.shape[-1]
        if self.debug:
            print("=" * 50 + " Shapes of our activations_big_flat (filtered yes cat) PTs " + "=" * 50) # DEBUG; fmt: skip
            print(activations_big_flat_pt.shape)  # DEBUG
            print(activations_small_flat_pt.shape)  # DEBUG
            print("=" * 100)  # DEBUG

        #### GET THE DOT PRODUCTS ####
        projs_big, projs_small = None, None
        if isinstance(projection, nn.Module):
            projs_big = projection(activations_big_flat_pt)
            projs_small = projection(activations_small_flat_pt)
        else:
            assert isinstance(projection, torch.Tensor)
            dot_patt = "b d, p d -> b p"  # batch proj dim
            dot_big = einops.einsum(activations_big_flat_pt, projection, dot_patt)
            dot_small = einops.einsum(activations_small_flat_pt, projection, dot_patt)
            assert dot_big.ndim == dot_small.ndim == 2
            assert dot_big.shape[0] >= len(self.big_sentences_str)  # at least one tok
            assert dot_small.shape[0] >= len(self.small_sentences_str)  # at least one tok
            assert dot_big.shape[-1] == dot_small.shape[-1] == 2  # only 2D supp.

            #### NORMALIZE THE DOT PRODUCTS ####
            # remember proj_{a onto b} = (a dot b) / |b|^2 * b => but we can ignore this last b
            # since we merely want the size on the b
            #
            # Project onto the two rows of the matrix
            l2s = torch.sum(projection.pow(2), dim=-1).squeeze()
            assert l2s.ndim == 1  # should be (2, )
            assert l2s.shape[0] == 2
            # https://pytorch.org/docs/stable/notes/broadcasting.html
            projs_big = dot_big / l2s
            projs_small = dot_small / l2s
        assert projs_big.ndim == projs_small.ndim == 2
        assert projs_big.shape[-1] == projs_small.shape[-1] == 2
        return projs_big, projs_small

    def plot_2d_projections(
        self,
        projections: Tuple[torch.Tensor, torch.Tensor],
        title: str,
        save_to: Path | str | None = None,
        x_left: Optional[float] = -10.0,
        x_right: Optional[float] = 10.0,
        y_bottom: Optional[float] = -10.0,
        y_top: Optional[float] = 10.0,
    ):

        # Extract x and y coordinates from projections
        x_big = projections[0][:, 0].detach().cpu().numpy()
        y_big = projections[0][:, 1].detach().cpu().numpy()
        x_small = projections[1][:, 0].detach().cpu().numpy()
        y_small = projections[1][:, 1].detach().cpu().numpy()

        assert (x_left is None or x_left <= x_big.min()) and (x_right is None or x_big.max() <= x_right), f"x_big: {x_big.min():4e}, {x_big.max():4e} (pick bigger frame)" # fmt: skip
        assert (y_bottom is None or y_bottom <= y_big.min()) and (y_top is None or y_big.max() <= y_top), f"y_big: {y_big.min():4e}, {y_big.max():4e} (pick bigger frame)" # fmt: skip
        assert (x_left is None or x_left <= x_small.min()) and (x_right is None or x_small.max() <= x_right), f"x_small: {x_small.min():4e}, {x_small.max():4e} (pick bigger frame)" # fmt: skip
        assert (y_bottom is None or y_bottom <= y_small.min()) and (y_top is None or y_small.max() <= y_top), f"y_small: {y_small.min():4e}, {y_small.max():4e} (pick bigger frame)" # fmt: skip

        # Create a scatter plot
        plt.figure(figsize=(10, 10))
        plt.scatter(x_big, y_big, label="Big Words", alpha=0.6, color="red")
        plt.scatter(x_small, y_small, label="Small Words", alpha=0.6, color="blue")
        plt.title(title)
        plt.xlim(x_left, x_right)
        plt.ylim(y_bottom, y_top)
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        plt.legend()
        if save_to is not None:
            assert not Path(save_to).exists(), f"File {save_to} already exists"
            plt.savefig(save_to)
        else:
            plt.show()
        plt.close()

    def get_and_plot_2d_projections(
        self,
        model: torch.nn.Module,
        projection: torch.Tensor | nn.Module,
        hook_name: str,
        use_post_hook: bool = False,
        projection_name: str = "",
        save_to: Path | str | None = None,
    ):
        """
        Combines the two above methods if you don't want to call them separately.
        """
        proj_big, proj_small = self.get_2d_projections(
            model, projection, hook_name, use_post_hook
        )
        self.plot_2d_projections(
            (proj_big, proj_small),
            f"Big and Small Words on projection {projection_name}",
            save_to=save_to,
        )


if __name__ == "__main__":
    litmus_tester = BigSmallLitmusTester(tokenizer="gpt2", debug=True)
    device = "cuda"
    model = HookedTransformer.from_pretrained("gpt2")
    model = model.to(device)
    d_model = 768
    projection = nn.Linear(d_model, 2).to(device)
    print("TRYING POST @ TL HOOKED MODEL")
    projections = litmus_tester.get_2d_projections(
        model, projection, "blocks.6.hook_resid_pre", use_post_hook=True
    )
    hf_model = AutoModelForCausalLM.from_pretrained("gpt2")
    hf_model = hf_model.to(device)
    print("TRYING POST @ HF MODEL")
    litmus_tester.get_2d_projections(
        hf_model, projection, "transformer.h.6", use_post_hook=True # Post
    )
    print("TRYING PRE @ HF MODEL")
    litmus_tester.get_2d_projections(
        hf_model, projection, "transformer.h.6", use_post_hook=False # Pre
    )
    print("PROJECTIONS SHAPESSSS")
    print(projections[0].shape)  # XXX debug
    print(projections[1].shape)  # XXX debug
    litmus_tester.plot_2d_projections(projections, "Big and Small Words", save_to="big_small_plot.png")  # Display
