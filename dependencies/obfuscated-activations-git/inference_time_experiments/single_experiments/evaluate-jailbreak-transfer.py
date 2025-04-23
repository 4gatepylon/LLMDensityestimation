import pickle
import time
from pathlib import Path
from typing import List, Tuple

import hydra
import numpy as np
import torch
import tqdm
from matplotlib import pyplot as plt
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from evaluate import evaluate_metric, evaluate_model, evaluate_model_and_metric_harmful
from obf_reps.config import ExperimentConfig
from obf_reps.data import ObfusDataModule, ObfusDataset
from obf_reps.logging import Logger
from obf_reps.metrics import ObfMetric
from obf_reps.models import ModelBase
from obf_reps.optimize.loss import LossFunctionBase
from obf_reps.optimize.optimizers import OptimizerBase
from obf_reps.plotting import plot_lat_scan
from obf_reps.utils import validate_and_create_experiment_config


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    experiment_cfg = validate_and_create_experiment_config(cfg)

    # Log information about the dataset
    experiment_cfg.logger.log(
        {"concept_dataset_train_size": len(experiment_cfg.concept_data_module.train_dataset)}
    )
    experiment_cfg.logger.log(
        {"concept_dataset_val_size": len(experiment_cfg.concept_data_module.val_dataset)}
    )
    experiment_cfg.logger.log(
        {"concept_dataset_test_size": len(experiment_cfg.concept_data_module.test_dataset)}
    )

    assert experiment_cfg.load_probe_path is not None
    assert experiment_cfg.load_pca_path is not None
    assert experiment_cfg.load_tunable_params_path is not None

    # Load up those tunable parameters!
    with open(experiment_cfg.load_tunable_params_path, "rb") as f:
        loaded_params = pickle.load(f)
    experiment_cfg.optimizer.model.tunable_params = loaded_params
    print(f"Loaded tunable parameters from {experiment_cfg.load_tunable_params_path}")

    # Evaluate the metric on those bad lads
    evaluate_model_and_metric_harmful(
        model=experiment_cfg.optimizer.model,
        obfus_data_module=experiment_cfg.obfus_data_module,
        concept_data_module=experiment_cfg.concept_data_module,
        metric=experiment_cfg.metric,
        logger=experiment_cfg.logger,
        at_def_epoch=1,
        who_just_went="attacker",
        gen_len=experiment_cfg.gen_len,
        params_last_train_epoch=0,
        evaluate_metric_on_gens=False,
        evaluate_behavior=False,
    )


if __name__ == "__main__":
    main()
