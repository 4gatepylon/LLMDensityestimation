from pathlib import Path

import pytest
import torch
from transformers import LlamaForCausalLM

from obf_reps.models.llama3 import Llama3SoftPrompted, ModelConfig, ModelReturn

pytestmark = pytest.mark.slow  # Marks all tests in this file as slow


@pytest.fixture(scope="module")
def model_path():
    raise ValueError("Specify path to llama-3-8b-below")
    return Path("paht/to/llama-3-8b/")


@pytest.fixture(scope="module")
def model_config():
    return ModelConfig()


@pytest.fixture(scope="module")
def llama_soft_prompted(model_path, model_config) -> Llama3SoftPrompted:
    return Llama3SoftPrompted.load_model(model_path, model_config)


class TestLlama3SoftPrompted:
    vocab_size = 128256
    hidden_size = 4096
    num_layers = 33

    def test_tokenize(self, llama_soft_prompted: Llama3SoftPrompted):
        text = "Hello, world!"
        input_ids, attention_mask = llama_soft_prompted.tokenize(text, add_chat_template=False)

        assert isinstance(input_ids, torch.Tensor)
        assert isinstance(attention_mask, torch.Tensor)
        assert input_ids.shape == attention_mask.shape
        assert input_ids.dim() == 2
        assert attention_mask.dim() == 2

    def test_tokenize_batch(self, llama_soft_prompted: Llama3SoftPrompted):
        text = ["Hello, world!", "This is a test sentence.", "Another test sentence."]
        input_ids, attention_mask = llama_soft_prompted.tokenize(text, add_chat_template=False)

        assert isinstance(input_ids, torch.Tensor)
        assert isinstance(attention_mask, torch.Tensor)
        assert input_ids.shape == attention_mask.shape
        assert input_ids.dim() == 2
        assert attention_mask.dim() == 2
        assert input_ids.shape[0] == attention_mask.shape[0] == len(text)

    def test_to_string(self, llama_soft_prompted: Llama3SoftPrompted):
        input_ids = torch.randint(0, 1000, (1, 10))
        result = llama_soft_prompted.to_string(input_ids)

        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], str)

    def test_init_random_tunable_params(self, llama_soft_prompted: Llama3SoftPrompted):
        params = llama_soft_prompted.tunable_params

        assert isinstance(params, torch.nn.ParameterList)
        assert len(params) == 1
        assert isinstance(params[0], torch.nn.Parameter)
        assert params[0].shape == (1, 100, self.hidden_size)
        assert params[0].requires_grad

    def test_init_prompt_tunable_params(self, model_path):
        prompt_init = "This is a test sentence."
        llama_soft_prompted = Llama3SoftPrompted.load_model(
            model_path, ModelConfig(prompt_init=prompt_init)
        )
        tokenized_prompt, _ = llama_soft_prompted.tokenize(prompt_init, add_special_tokens=False)
        params = llama_soft_prompted.get_tunable_params()

        assert isinstance(params, torch.nn.ParameterList)
        assert len(params) == 1
        assert isinstance(params[0], torch.nn.Parameter)
        assert params[0].shape == (1, 6, self.hidden_size)
        assert params[0].requires_grad
        assert torch.allclose(
            params[0],
            llama_soft_prompted.model.get_input_embeddings()(tokenized_prompt),
        )

    def test_tokenize_without_special_tokens(self, llama_soft_prompted: Llama3SoftPrompted):
        text = "Hello, world!"
        input_ids, attention_mask = llama_soft_prompted.tokenize(text, add_special_tokens=False)

        assert isinstance(input_ids, torch.Tensor)
        assert isinstance(attention_mask, torch.Tensor)
        assert input_ids.shape == attention_mask.shape
        assert input_ids.dim() == 2
        assert attention_mask.dim() == 2
        assert llama_soft_prompted.tokenizer.bos_token_id not in input_ids

    def test_tokenize_with_max_length(self, llama_soft_prompted: Llama3SoftPrompted):
        text = "This is a very long sentence that should be truncated."
        max_length = 5
        input_ids, attention_mask = llama_soft_prompted.tokenize(
            text, add_special_tokens=True, max_length=max_length
        )

        assert input_ids.shape[1] == max_length
        assert attention_mask.shape[1] == max_length

    def test_forward_from_ids_without_target(self, llama_soft_prompted: Llama3SoftPrompted):
        input_ids, attention_mask = llama_soft_prompted.tokenize(
            "Hello, world!", add_special_tokens=False
        )
        result = llama_soft_prompted.forward_from_ids(input_ids, attention_mask)

        assert isinstance(result, ModelReturn)
        assert isinstance(result.logits, torch.Tensor)
        assert isinstance(result.reps, torch.Tensor)
        assert result.logits.shape == (1, 1, self.vocab_size)
        assert result.reps.shape == (1, self.num_layers, 1, self.hidden_size)
        assert result.loss is None
        assert result.loss_mask is None

    def test_forward_from_ids_with_target(self, llama_soft_prompted: Llama3SoftPrompted):
        input_ids, input_attention_mask = llama_soft_prompted.tokenize(
            "Hello, world!", add_special_tokens=False
        )
        target_ids, target_attention_mask = llama_soft_prompted.tokenize(
            "This is a target.", add_special_tokens=False
        )
        result = llama_soft_prompted.forward_from_ids(
            input_ids, input_attention_mask, target_ids, target_attention_mask
        )

        assert isinstance(result, ModelReturn)
        assert isinstance(result.logits, torch.Tensor)
        assert isinstance(result.reps, torch.Tensor)
        assert isinstance(result.loss, torch.Tensor)
        assert isinstance(result.loss_mask, torch.Tensor)
        assert result.logits.shape[0] == 1
        assert result.logits.shape[2] == self.vocab_size
        assert result.reps.shape == (
            1,
            self.num_layers,
            target_ids.shape[1],
            self.hidden_size,
        )

    def test_forward_from_ids_with_and_without_tunable_params(
        self, llama_soft_prompted: Llama3SoftPrompted
    ):
        input_ids, attention_mask = llama_soft_prompted.tokenize(
            "Hello, world!", add_special_tokens=False
        )

        result_with_params = llama_soft_prompted.forward_from_ids(
            input_ids, attention_mask, use_tunable_params=True
        )
        result_without_params = llama_soft_prompted.forward_from_ids(
            input_ids, attention_mask, use_tunable_params=False
        )

        assert not torch.allclose(result_with_params.logits, result_without_params.logits)
        assert not torch.allclose(result_with_params.reps, result_without_params.reps)


def test_load_model(model_path, model_config):
    llama_soft_prompted = Llama3SoftPrompted.load_model(model_path, model_config)

    assert isinstance(llama_soft_prompted, Llama3SoftPrompted)
    assert isinstance(llama_soft_prompted.model, LlamaForCausalLM)
