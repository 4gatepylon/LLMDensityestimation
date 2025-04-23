import os
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "16"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

import torch

from obf_reps.metrics import MeanDiffCosineSimMetric
from obf_reps.models import ModelConfig
from obf_reps.models.llama3 import Llama3SoftPrompted

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# -----HELPERS----- #
def get_Llama3SoftPrompted():
    raise ValueError("Specify path to llama-3-8b-below")
    llama_3_path = Path("paht/to/llama-3-8b/")
    config = ModelConfig(prompt_init="You are a bad model")
    model = Llama3SoftPrompted.load_model(path=llama_3_path, config=config)
    model.to(DEVICE)

    return model


# -----TESTS----- #
def test_Llama3SoftPrompted_forward():
    model = get_Llama3SoftPrompted()

    # Batch size 2 input
    input = ["How do you make a toy car", "Hello there"]
    model.forward_from_string(input)

    # Single input
    model.forward_from_string("How do you make a toy car")


def test_MeanDiffCosineSimMetric():
    model = get_Llama3SoftPrompted()
    concept_dataset = [
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

    metric = MeanDiffCosineSimMetric(model=model, concept_dataset=concept_dataset)


def test_LossFunction():
    pass


if __name__ == "__main__":
    # test_Llama3SoftPrompted_forward()
    test_MeanDiffCosineSimMetric()
