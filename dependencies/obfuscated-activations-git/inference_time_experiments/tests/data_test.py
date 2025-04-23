import pytest

from obf_reps.config import ExperimentConfig
from obf_reps.data import (
    HarmfulConceptDataModule,
    JailbreakObfusDataModule,
    TruthfulConceptDataModule,
)


@pytest.fixture
def config():
    return ExperimentConfig(
        model_path="model_path",
        concept_data_path="concept_data_path",
        obfus_data_path="obfus_data_path",
    )


@pytest.mark.slow
def test_harmful_concept_data_module(config: ExperimentConfig):

    data_module = HarmfulConceptDataModule(config)
    data_module.load_dataset()
    dataset = data_module.dataset

    assert isinstance(dataset, list)
    for item in dataset:
        assert len(item) == 2


@pytest.mark.slow
@pytest.mark.skip(reason="Not implemented yet")
def test_truthful_concept_data_module(config: ExperimentConfig):

    data_module = TruthfulConceptDataModule(config)
    data_module.load_dataset()
    dataset = data_module.dataset

    assert isinstance(dataset, list)
    for item in dataset:
        assert len(item) == 2
