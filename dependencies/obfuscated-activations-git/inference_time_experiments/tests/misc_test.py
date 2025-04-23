from obf_reps.config import ExperimentConfig


def test_config():
    _ = ExperimentConfig(
        model_path="model_path",
        concept_data_path="concept_data_path",
        obfus_data_path="obfus_data_path",
    )
