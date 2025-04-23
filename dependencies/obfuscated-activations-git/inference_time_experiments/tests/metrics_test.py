from pathlib import Path

import pytest
import torch

from obf_reps.metrics import MeanDiffCosineSimMetric
from obf_reps.models.llama3 import Llama3SoftPrompted, ModelConfig

batch_size = 2
num_layers = 3
seq_len = 1
hidden_dim = 10


@pytest.fixture
def reps():
    return torch.rand(batch_size, num_layers, seq_len, hidden_dim)


@pytest.fixture
def dummy_model():
    class DummyModel:
        def forward_from_string(self, text, **kwargs):

            if isinstance(text, str):
                batch_size = 1
            elif isinstance(text, list):
                batch_size = len(text)
            else:
                raise ValueError("Input text must be a string or a list of strings")

            return type(
                "obj",
                (object,),
                {"reps": torch.rand(batch_size, num_layers, seq_len, hidden_dim)},
            )

    return DummyModel()


@pytest.fixture(scope="module")
def model_path():
    raise ValueError("Specify path to llama-3-8b-below")
    return Path("paht/to/llama-3-8b/")


@pytest.fixture(scope="module")
def model_config():
    return ModelConfig(model_dtype=torch.float32)


@pytest.fixture(scope="module")
def llama_soft_prompted(model_path, model_config):
    return Llama3SoftPrompted.load_model(model_path, model_config)


@pytest.fixture
def concept_dataset():
    return [
        ("blue", "red"),
        ("sky", "apple"),
        ("ocean", "strawberry"),
        ("blueberry", "cherry"),
        ("sapphire", "ruby"),
        ("blue jay", "cardinal"),
        ("denim", "tomato"),
        ("blueprint", "stop sign"),
        ("blue whale", "lobster"),
        ("police light", "fire truck"),
        ("peacock feather", "rose"),
    ]


class TestMetricsWithDummy:
    def test_mean_diff_cosine_sim_metric(self, reps, dummy_model):
        metric = MeanDiffCosineSimMetric(dummy_model, [("positive", "negative")])
        result = metric.evaluate(reps)

        assert isinstance(result, torch.Tensor)
        assert result.shape == torch.Size([batch_size, num_layers, seq_len])


@pytest.mark.slow
class TestMetricsWithLlama3:
    num_layers = 33

    def test_mean_diff_cosine_sim_metric(self, llama_soft_prompted, concept_dataset):
        metric = MeanDiffCosineSimMetric(llama_soft_prompted, concept_dataset)

        input_text = "This is a test sentence."
        result = llama_soft_prompted.forward_from_string(input_text)
        metric_result = metric.evaluate(result.reps)

        assert isinstance(metric_result, torch.Tensor)
        assert metric_result.shape == (1, self.num_layers, 1)

    @pytest.mark.skip(reason="Flaky")
    def test_metric_difference(self, llama_soft_prompted, concept_dataset):
        mean_diff_metric = MeanDiffCosineSimMetric(llama_soft_prompted, concept_dataset)

        positive_text = "Bluejeans"
        negative_text = "Ladybug"

        positive_result = llama_soft_prompted.forward_from_string(
            positive_text, use_tunable_params=False
        )
        negative_result = llama_soft_prompted.forward_from_string(
            negative_text, use_tunable_params=False
        )

        mean_diff_pos = mean_diff_metric.evaluate(positive_result.reps)
        mean_diff_neg = mean_diff_metric.evaluate(negative_result.reps)

        assert torch.all(mean_diff_pos != mean_diff_neg)

    def test_metrics_shape_consistency(self, llama_soft_prompted, concept_dataset):
        mean_diff_metric = MeanDiffCosineSimMetric(llama_soft_prompted, concept_dataset)

        input_texts = [
            "Short text.",
            "This is a positive sentence.",
            "This is a negative sentence.",
            "Red",
            "This is a longer sentence that ends with the word 'blue'.",
        ]

        for text in input_texts:
            result = llama_soft_prompted.forward_from_string(text)
            mean_diff_result = mean_diff_metric.evaluate(result.reps)

            assert isinstance(mean_diff_result, torch.Tensor)

            assert mean_diff_result.shape == (1, self.num_layers, 1)
