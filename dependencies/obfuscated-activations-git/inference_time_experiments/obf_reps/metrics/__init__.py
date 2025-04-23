import gc
import json
import os
import pickle
import random
import sys
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, TypeVar

import joblib
import numpy as np
import torch
import tqdm
from jaxtyping import Bool, Float
from sae_lens import SAE  # For SAEs for Gemma-2-2b
from sklearn.decomposition import PCA
from sparsify import Sae  # For Eleuther SAEs for Llama-3-8b
from torch import Tensor

from obf_reps.data import ConceptDataModule, ConceptDataset, ObfusDataset
from obf_reps.data.data_utils import join_concept_obfus_datasets
from obf_reps.logging import Logger
from obf_reps.metrics.probes import MLP, VAE, LogisticRegression, SAEClassifier
from obf_reps.models import ForwardReturn, ModelBase

DEBUG = False

RepsDatasetT = List[Tuple[Float[Tensor, "1 layers red_seq_len h_dim"], Float[Tensor, ""]]]
CachedRepsT = TypeVar("CachedRepsT", bound="CachedReps")

MemRepsDatasetT = Tuple[np.memmap, np.memmap]


# Helper functions for measuring Cached representation sizes
def get_tensor_size_in_gb(tensor: torch.Tensor) -> float:
    return tensor.element_size() * tensor.nelement() / (1024**3)


def get_dataset_size_in_gb(
    dataset: List[Tuple[Float[Tensor, "1 layers red_seq_len h_dim"], Float[Tensor, ""]]]
) -> float:
    total_size = 0
    for rep, label in dataset:
        total_size += get_tensor_size_in_gb(rep)
        total_size += get_tensor_size_in_gb(label)
    return total_size


# We cache reps in two ways. The first stores a RepsDataset, the other just a list of tensorts.
# This could be consolidated.
@dataclass
class CachedReps:
    attack_defence_epoch: int
    rep_dataset: RepsDatasetT
    who_just_went: str

    @staticmethod
    def combine(cached_reps: List[CachedRepsT]) -> RepsDatasetT:

        combined_dataset = []
        for cache in cached_reps:
            combined_dataset.extend(cache.rep_dataset)
        # Shuffle the dataset
        random.shuffle(combined_dataset)
        return combined_dataset


@dataclass
class TestCachedReps:
    label: str  # benign
    attack_defence_epoch: int
    reps: List[Tensor]


@dataclass
class MetricConfig:
    layers: List[int]  # Which layers to probe

    n_components: Optional[int] = 1  # Number of components for PCA
    feature_list_filepath: Optional[str] = None  # Path to feature list for Gemma-2-2b
    results_dir: Optional[str] = (
        None  # Path where evaluation results on the eval obfuscation dataset will be saved
    )


def transfer_to_cpu(tensor_pair):
    rep, label = tensor_pair
    return (rep.to("cpu"), label.to("cpu"))


class ObfMetric(ABC):
    def __init__(
        self,
        model: ModelBase,
        data_module: ConceptDataModule,
        config: MetricConfig,
        logger: Logger,
        probe_path: Optional[str] = None,
        pca_path: Optional[str] = None,
    ):
        """Base class for metrics used to detect binary concepts from model activations.

        Args:
            model: The model to extract representations from
            data_module: Data module containing training and test data
            config: Configuration specifying which layers to probe and other settings
            logger: Logger for tracking metrics and results
            probe_path: Path to load pretrained probe weights from
            pca_path: Path to load pretrained PCA components from

        The metric works by:
        1. Extracting representations from the model for training data
        2. Fitting a probe (if not loading pretrained) to distinguish obfuscated from benign examples
        3. Training PCA (if not loading pretrained) [which is only used for visualization plots]
        4. Caching representations in reps_bank for reuse later

        Subclasses must implement:
        - load_pretrained(): Load a pretrained probe
        - fit(): Train the probe on representations
        - evaluate(): Apply the probe to get raw scores
        - predict(): Convert raw scores to final predictions
        """

        super().__init__()
        self.model = model
        self.device = self.model.device
        self.model_hidden_dim = self.model.model.config.hidden_size
        self.model_num_layers = self.model.model.config.num_hidden_layers

        self.data_module = data_module
        self.config = config
        self.logger = logger

        self.train_reps_reduce = data_module.train_reps_reduce
        self.test_reps_reduce = data_module.test_reps_reduce

        # Stores previously computed reps
        self.reps_bank: List[CachedReps] = []
        self.test_reps_bank: List[CachedReps] = []

        if probe_path and pca_path:
            self.load_pretrained(probe_path)
            self.load_pca(pca_path)
        else:

            # In this case we need the train_rep_dataset
            print("Getting train reps")
            train_rep_dataset = self.get_reps_from_dataset(
                model,
                self.data_module.train_dataset,  # type: ignore
                self.train_reps_reduce,
            )
            print("Done getting train reps")

            # Cache reps
            cached_reps = CachedReps(0, train_rep_dataset, "start")
            self.reps_bank.append(cached_reps)

            if probe_path is None:
                self.fit(train_rep_dataset)
            else:
                self.load_pretrained(probe_path)
            if pca_path is None:
                self.pca = self.train_pca(train_rep_dataset)
            else:
                self.load_pca(pca_path)

    @abstractmethod
    def load_pretrained(
        self,
        path: Optional[str] = None,
        probe_state: Optional[Any] = None,
    ):
        """Load a pretrained probe from a path or probe state.

        Args:
            path (Optional[str]): Path to load the probe state from. If provided, the probe state will be loaded from this path.
            probe_state (Optional[Any]): Probe state to load directly. If provided, the probe will be loaded from this state.

        Note:
            One of path or probe_state must be provided.
        """
        ...

    @abstractmethod
    def save_probe(
        self,
        path: Optional[str] = None,
    ) -> None | Any:
        """Save the probe state to a path or return it.

        Args:
            path (Optional[str]): Path to save the probe state to. If provided, the probe state will be saved to this path.

        Returns:
            Optional[Any]: If path is None, returns the probe state. Otherwise returns None after saving to path.
        """
        ...

    def train_pca(self, train_rep_dataset: RepsDatasetT) -> PCA:
        """Train a PCA model on the representations from the training dataset.

        Args:
            train_rep_dataset (RepsDatasetT): Dataset containing model representations and labels.

        Returns:
            PCA: Fitted PCA model that reduces dimensionality to 2 components.

        Note:
            This method also sets self.mean, self.std and self.pca_layer_idx as side effects.
            The PCA is trained on representations from a single layer (2/3 through the network)
            after normalizing the data.
        """

        # Extract all representations and concatenate them
        all_reps = []
        example_rep = train_rep_dataset[0][0]
        n_layers = example_rep.shape[1]
        self.pca_layer_idx = int(n_layers * 2 / 3)  # Select 2/3 of the way through layers

        for rep, _ in train_rep_dataset:
            # Extract representations from the selected layer
            layer_rep = rep[0, self.pca_layer_idx]  # Shape: [seq_len, h_dim]

            # Treat each vector along seq_len as independent
            all_reps.append(layer_rep)

        # Concatenate all representations
        all_reps = torch.cat(all_reps, dim=0)  # Shape: [total_seq_len, h_dim]

        # Convert to numpy for sklearn PCA
        all_reps_np = all_reps.cpu().numpy()

        # Normalize the data
        self.mean = np.mean(all_reps_np, axis=0)
        self.std = np.std(all_reps_np, axis=0)
        all_reps_normalized = (all_reps_np - self.mean) / (self.std + 1e-8)

        # Sanity check: Calculate and print mean and standard deviation of normalized data
        normalized_mean = np.mean(all_reps_normalized, axis=0)
        normalized_std = np.std(all_reps_normalized, axis=0)

        # Fit PCA on normalized data
        pca = PCA(n_components=2)
        pca.fit(all_reps_normalized)

        return pca

    def predict_pca(self, reps: List[Float[Tensor, "1 layers seq_len h_dim"]]) -> np.ndarray:
        """Apply PCA transformation to the input representations.

        Args:
            reps: List of tensors, each of shape [1, layers, seq_len, h_dim]

        Returns:
            np.ndarray: 2D array of shape [num_reps * seq_len, 2] containing PCA-transformed points
        """
        all_reps = []

        for rep in reps:
            # Extract representations from the selected layer
            layer_rep: Float[Tensor, "seq_len h_dim"] = rep[0, self.pca_layer_idx]

            # Treat each vector along seq_len as independent
            all_reps.append(layer_rep)

        # Concatenate all representations
        all_reps: Float[Tensor, "total_seq_len h_dim"] = torch.cat(all_reps, dim=0)

        # Convert to numpy
        all_reps_np = all_reps.cpu().numpy()

        # Normalize the data using the same mean and std from training
        all_reps_normalized = (all_reps_np - self.mean) / (self.std + 1e-8)

        # Apply PCA transformation
        pca_transformed: np.ndarray = self.pca.transform(all_reps_normalized)

        return pca_transformed

    @abstractmethod
    def fit(
        self,
        reps_dataset: RepsDatasetT,
        **kwargs,
    ):
        """Fit the metric to the data.

        Args:
            reps_dataset: Dataset containing model representations and labels.
        """
        ...

    @abstractmethod
    def evaluate(
        self,
        reps: Float[Tensor, "b layers seq_len hidden_dim"],
        **kwargs,
    ) -> Float[Tensor, "b layers seq_len"]:
        """Computes obfuscation metric for every representation of shape [hidden_dim] in the input.
        Thus evaluate collapses the hidden_dim dimension of the input.

        Args:
            reps: Input representations of shape [b, layers, seq_len, hidden_dim].

        Returns:
            Float[Tensor, "b layers seq_len"]: Obfuscation metric scores for each representation.
        """
        ...

    def predict(
        self,
        reps: Float[Tensor, "b layers seq_len hidden_dim"],
        attention_mask: Optional[Bool[Tensor, "b seq_len"]] = None,
        layer_reduction: str = "mean",
        **kwargs,
    ) -> Float[Tensor, "b"]:
        """Predict the labels of the input using the metric score.

        Note that this is different to evaluate as it predicts a single score for each element of
        the input.

        Args:
            reps: reps to predict.
            layer_reduction: how to aggregate scores over layers.
            attention_mask: boolean mask, 0 when reps correspond to <pad> tokens.
        """
        ...

    @torch.no_grad()
    def get_reps_from_dataset(
        self,
        model: ModelBase,
        dataset: ConceptDataset,
        reps_reduce: Callable[[Tensor, Tensor], Tensor],
        use_tunable_params: bool = False,
    ) -> RepsDatasetT:
        """Convert a dataset to info needed to train and evaluate a metric.

        To train and evaluate a metric, we need to take each example in the
        dataset and convert them into:

        - the reps that should be fed to into a metric
        - the label for the given example

        For a text input x_i, reps_i could have different shapes depending
        on the metric and task (in some cases the task may demand you look
        at a single rep, enforced by reps_reduce), so we store
        the results in a list.

        Args:
            model: The model to extract representations from
            dataset: The dataset to convert to reps
            reps_reduce: The function to reduce the reps to the desired shape, taken from ConceptDataModule
            use_tunable_params: Whether to use tunable parameters in the model forward pass

        Returns:
            RepsDatasetT: A list of tuples, where each tuple contains a representation and its corresponding label.
        """

        positive_reps, negative_reps = [], []

        reps_dataset = []
        pos_target_len = 0
        neg_target_len = 0

        i = 0
        timings = {
            "positive_forward": 0.0,
            "negative_forward": 0.0,
            "reduce_and_process": 0.0,
            "cleanup": 0.0,
            "cpu_transfer": 0.0,
            "garbage_collect": 0.0,
            "clear_gpu_cache": 0.0,
        }

        for i, ((pos_input, pos_target), (neg_input, neg_target)) in enumerate(tqdm.tqdm(dataset)):
            # Time positive forward pass
            t0 = time.perf_counter()
            positive_rep: ForwardReturn = model.forward_from_string(
                input_text=pos_input,  # type: ignore
                target_text=pos_target,
                add_chat_template=True,
                use_tunable_params=use_tunable_params,
            )
            timings["positive_forward"] += time.perf_counter() - t0

            # Time negative forward pass
            t0 = time.perf_counter()
            negative_rep: ForwardReturn = model.forward_from_string(
                input_text=neg_input,  # type: ignore
                target_text=neg_target,
                add_chat_template=True,
                # This is false because we want the standard model forward on the "benign" examples
                use_tunable_params=False,
            )
            timings["negative_forward"] += time.perf_counter() - t0

            # Time reduce and processing
            t0 = time.perf_counter()
            pos_input_reps, pos_target_reps, pos_target_mask = (
                positive_rep.input_reps,
                positive_rep.target_reps,
                positive_rep.loss_mask,
            )
            neg_input_reps, neg_target_reps, neg_target_mask = (
                negative_rep.input_reps,
                negative_rep.target_reps,
                negative_rep.loss_mask,
            )

            pos_target_len += pos_target_reps.shape[2]
            neg_target_len += neg_target_reps.shape[2]

            pos_rep: Float[Tensor, "1 layers red_seq_len_1 h_dim"]
            pos_rep, _ = reps_reduce(pos_input_reps, pos_target_reps, pos_target_mask)
            neg_rep: Float[Tensor, "1 layers red_seq_len_2 h_dim"]
            neg_rep, _ = reps_reduce(neg_input_reps, neg_target_reps, neg_target_mask)

            timings["reduce_and_process"] += time.perf_counter() - t0

            t0 = time.perf_counter()
            pos_rep = pos_rep.detach().to("cpu")
            neg_rep = neg_rep.detach().to("cpu")
            timings["cpu_transfer"] += time.perf_counter() - t0

            reps_dataset.append(
                (pos_rep, torch.tensor([1.0], device="cpu")),
            )
            reps_dataset.append(
                (neg_rep, torch.tensor([0.0], device="cpu")),
            )

            if DEBUG and (i + 1) % 10 == 0:
                print(f"Cached {i + 1} examples")
                print(f"Cached {get_dataset_size_in_gb(reps_dataset)} GB of reps")

            # Time cleanup
            t0 = time.perf_counter()
            del pos_input_reps, pos_target_reps, pos_target_mask
            del neg_input_reps, neg_target_reps, neg_target_mask
            del _
            timings["cleanup"] += time.perf_counter() - t0

            t0 = time.perf_counter()
            torch.cuda.empty_cache()
            timings["clear_gpu_cache"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        timings["garbage_collect"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        torch.cuda.empty_cache()
        timings["clear_gpu_cache"] += time.perf_counter() - t0

        print(f"Cached {i + 1} examples")
        print(f"Cached {get_dataset_size_in_gb(reps_dataset)} GB of reps")
        print(f"Get Reps Timings: \n{timings}")
        print(f"Dataset len: {len(dataset)}")

        print(f"Average pos target len: {pos_target_len/(i+1)}")
        print(f"Average neg target len: {neg_target_len/(i+1)}")

        random.shuffle(reps_dataset)

        return reps_dataset

    def get_reps(
        self,
        input_text: str,
        target_text: str,
        reps_reduce: Callable[[Tensor, Tensor], Tensor],
        use_tunable_params: bool = True,
    ) -> Float[Tensor, "1 layers red_seq_len h_dim"]:
        """Get reps from a single input and target from the model.

        Args:
            input_text: The input text to get representations for
            target_text: The target text to get representations for
            reps_reduce: Function that reduces the representations from the model
            use_tunable_params: Whether to use tunable parameters in the model forward pass

        Returns:
            Tensor of shape (1, layers, reduced_seq_len, hidden_dim) containing the reduced representations
        """

        rep: ForwardReturn = self.model.forward_from_string(
            input_text=input_text,
            target_text=target_text,
            add_chat_template=True,
            use_tunable_params=use_tunable_params,
        )
        input_reps, target_reps, target_mask = rep.input_reps, rep.target_reps, rep.loss_mask
        reduced_rep, _ = reps_reduce(input_reps, target_reps, target_mask)
        return reduced_rep.detach().to("cpu")

    def load_pca(self, path: str):
        """Load a trained PCA model from a pickle file.

        Args:
            path: Path to the pickle file containing the PCA model and associated data.
                 Expected to contain a dict with keys:
                 - 'pca': The fitted PCA model
                 - 'mean': Mean used for standardization
                 - 'std': Standard deviation used for standardization
                 - 'pca_layer_idx': Index of the layer the PCA was trained on

        Note:
            This method sets the self.pca, self.mean, self.std, and self.pca_layer_idx attributes.
        """

        with open(path, "rb") as f:
            pca_data = pickle.load(f)
        self.pca = pca_data["pca"]
        self.mean = pca_data["mean"]
        self.std = pca_data["std"]
        self.pca_layer_idx = pca_data["pca_layer_idx"]

    def save_pca(self, path: str):
        """Save a trained PCA model to a pickle file.

        Args:
            path: Path where the pickle file should be saved. The file will contain a dict with keys:
                 - 'pca': The fitted PCA model
                 - 'mean': Mean used for standardization
                 - 'std': Standard deviation used for standardization
                 - 'pca_layer_idx': Index of the layer the PCA was trained on
        """

        pca_data = {
            "pca": self.pca,
            "mean": self.mean,
            "std": self.std,
            "pca_layer_idx": self.pca_layer_idx,
        }
        with open(path, "wb") as f:
            pickle.dump(pca_data, f)

    def refit(self, attack_defense_epoch: int) -> None:
        """Refit the metric on new training data after an attack-defense epoch.

        This method collects new representations from the model after tuning, adds them to
        the cached representations bank, and refits the metric on the combined dataset.

        Args:
            attack_defense_epoch (int): The current epoch number in the attack-defense training loop.
                Used to verify the cached representations are properly ordered.

        Note:
            - Collects new representations using tunable parameters (attack mode)
            - Caches the new representations and combines with previous epochs
            - Refits the metric on the combined dataset
        """

        new_rep_dataset = self.get_reps_from_dataset(
            self.model,
            self.data_module.train_dataset,  # type: ignore
            self.train_reps_reduce,
            use_tunable_params=True,
        )

        # This is caching the train reps
        cached_reps = CachedReps(attack_defense_epoch, new_rep_dataset, "attacker")

        self.reps_bank.append(cached_reps)
        assert len(self.reps_bank) == attack_defense_epoch + 2
        new_train_dataset = CachedReps.combine(self.reps_bank)

        self.fit(new_train_dataset)

        torch.cuda.empty_cache()


class CosineSimMetric(ObfMetric, ABC):
    """Base class for metrics that use cosine similarity to measure obfuscation.

    CosineSimMetric computes obfuscation scores by measuring the cosine similarity between model
    representations and a learned direction vector. Subclasses must implement get_direction() to
    specify how this direction is determined.
    """

    @abstractmethod
    def get_direction(self) -> Float[Tensor, "layers hidden_dim"]:
        """Returns the direction vector used to compute cosine similarity scores.

        The direction vector represents the axis along which the target concept is measured.
        Cosine similarity between model representations and this direction indicates
        the degree of the concept.

        Returns:
            Float[Tensor, "layers hidden_dim"]: Direction vector for each layer, with
                shape [num_layers, hidden_dim]. Used to compute cosine similarity with
                model representations.
        """
        ...

    def evaluate(
        self,
        reps: Float[Tensor, "b layers seq_len hidden_dim"],
        **kwargs,
    ) -> Float[Tensor, "b layers seq_len"]:

        reps = reps.to(torch.float32)
        direction = self.get_direction().to(torch.float32)

        if reps.ndim == 3:
            # We got reps with no seq_len from "get_reps_from_dataset"
            # Add in sequence dimension to broadcast over
            reps = reps.unsqueeze(2)

        # Add in batch dimension and sequence dimension to broadcast over
        expanded_direction = direction.unsqueeze(0).unsqueeze(2)

        cosine_sim: Float[Tensor, "b layers seq_len"] = torch.cosine_similarity(
            reps, expanded_direction, dim=-1
        )

        return cosine_sim.to(torch.float16)

    def predict(
        self,
        reps: Float[Tensor, "b layers seq_len hidden_dim"],
        attention_mask: Optional[Bool[Tensor, "b seq_len"]] = None,
        layer_reduction: str = "mean",
        **kwargs,
    ) -> Float[Tensor, "b"]:
        """Scores are cosine similarities in [-1, 1]."""

        b, layers, seq_len, hidden_dim = reps.shape
        if b > 1:
            assert attention_mask is not None, "Attention mask must be provided for batch size > 1"

        scores: Float[Tensor, "b layers seq_len"] = self.evaluate(reps)

        # Apply attention mask if provided
        if attention_mask is not None:
            # Expand attention_mask to match scores dimensions
            attention_mask: Float[Tensor, "b layers seq_len"] = attention_mask.unsqueeze(
                1
            ).expand_as(scores)
            assert attention_mask.shape == scores.shape, "Bug in attention mask handling"
            scores = scores.masked_fill(~attention_mask, 0.0)

            # Sum scores and divide by the number of non-masked tokens
            seq_scores = scores.sum(dim=2) / attention_mask.sum(dim=2).clamp(min=1)
        else:
            seq_scores = scores.mean(dim=2)

        # Reduce over layers
        match layer_reduction:
            case "mean":
                final_scores = seq_scores.mean(dim=1)
            case "sum":
                final_scores = seq_scores.sum(dim=1)
            case "max":
                final_scores = seq_scores.max(dim=1).values
            case _:
                raise ValueError(f"Invalid layer_reduction: {layer_reduction}")

        return final_scores

    def load_pretrained(
        self,
        path: Optional[str] = None,
        probe_state: Optional[Any] = None,
    ):
        assert (
            path is not None or probe_state is not None
        ), "Must provide either path or probe_state"
        if path is not None:
            with open(path, "rb") as f:
                self.direction = pickle.load(f)
        else:
            assert isinstance(probe_state, torch.Tensor)
            self.direction = probe_state

    def save_probe(
        self,
        path: Optional[str] = None,
    ) -> None | Any:
        if path is not None:
            with open(path, "wb") as f:
                pickle.dump(self.direction, f)
            return None
        return self.direction


class MeanDiffCosineSimMetric(CosineSimMetric):
    """Metric that measures obfuscation using cosine similarity to the mean difference direction.

    This metric computes a direction vector by taking the difference between the mean
    representations of positive and negative examples in the training data.
    """

    def fit(
        self,
        reps_dataset: RepsDatasetT,
        **kwargs,
    ):

        device = self.device
        num_layers = self.model_num_layers + 1  # Data comes with embedding
        hidden_size = self.model_hidden_dim

        pos_sum = torch.zeros((num_layers, hidden_size), device=device)
        neg_sum = torch.zeros((num_layers, hidden_size), device=device)
        pos_count = 0
        neg_count = 0

        for rep, label in reps_dataset:

            _, layers, seq_len, hidden_dim = rep.shape
            rep = rep.squeeze(0).to(device)  # [layers, pos_seq_len, hidden_dim]

            # pos_rep and neg_rep shape: [1, layers, seq_len, hidden_dim]
            if label == 1:
                pos_sum += rep.sum(dim=1)  # [layers, hidden_dim]
                pos_count += seq_len
            else:
                neg_sum += rep.sum(dim=1)  # [layers, hidden_dim]
                neg_count += seq_len

        pos_mean = pos_sum / pos_count
        neg_mean = neg_sum / neg_count

        direction = (pos_mean - neg_mean).to(self.device)  # [layers, hidden_dim]
        norm = torch.norm(direction, dim=1, keepdim=True)
        self.direction = torch.where(norm != 0, direction / norm, torch.zeros_like(direction))

    def get_direction(self) -> Float[Tensor, "layers hidden_dim"]:
        return self.direction


class TrainableMetric(ObfMetric, ABC):
    """Base class for metrics that use trainable probes with gradient descent.

    This class provides common functionality for metrics that use neural network probes or other
    trainable models that require gradient-based optimization. Subclasses must implement
    create_model() to specify the probe architecture.

    The probe is trained on representations from each layer specified in config.layers. During
    training, each token's representation is treated as an independent example.
    """

    def __init__(
        self,
        model: ModelBase,
        data_module: ConceptDataModule,
        config: MetricConfig,
        logger: Logger,
        probe_path: Optional[str] = None,
        pca_path: Optional[str] = None,
    ):
        self.probe: Dict[int, torch.nn.Module] = {}
        super().__init__(model, data_module, config, logger, probe_path, pca_path)

    @abstractmethod
    def create_model(self, hidden_size) -> torch.nn.Module:
        """Create a trainable probe model.

        This abstract method must be implemented by subclasses to specify the probe architecture.
        The probe takes representations from a single layer as input and outputs predictions.

        Args:
            hidden_size (int): Dimension of the input representations

        Returns:
            torch.nn.Module: A PyTorch module implementing the probe architecture
        """
        ...

    def fit(
        self,
        reps_dataset: RepsDatasetT,
        **kwargs,
    ):
        device = self.device

        # Concatenate all reps and labels, treating each token as an independent example
        all_reps = []
        all_labels = []

        for rep, label in reps_dataset:
            # rep shape: [1, layers, seq_len, hidden_dim]
            # Reshape to [layers, seq_len, hidden_dim]
            rep = rep.squeeze(0)
            # Extend labels to match seq_len
            extended_label = label.repeat(rep.shape[1])
            all_reps.append(rep)
            all_labels.append(extended_label)

        # Concatenate along the sequence length dimension
        reps = torch.cat(all_reps, dim=1)
        labels = torch.cat(all_labels, dim=0)

        n_layers, total_seq_len, hidden_size = reps.shape

        for layer in self.config.layers:
            X_train = reps[layer]  # Shape: [total_seq_len, hidden_size]
            assert X_train.shape == (
                total_seq_len,
                hidden_size,
            ), f"Incorrect X_train shape: {X_train.shape}"

            y_train = labels

            model = self.create_model(hidden_size).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5, eps=1e-5)
            criterion = torch.nn.BCEWithLogitsLoss()

            dataloader = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(X_train, y_train),
                batch_size=64,
                shuffle=True,
            )

            print(f"Fitting layer {layer} probe")
            for epoch in range(1):
                for X, y in tqdm.tqdm(dataloader):
                    optimizer.zero_grad()
                    outputs = model(X.to(device)).squeeze()
                    loss = criterion(outputs, y.to(device))
                    loss.backward()
                    optimizer.step()
                    self.logger.log({f"probe_loss_layer_{layer}": loss.item()})

            self.probe[layer] = model

    def evaluate(
        self, reps: Float[Tensor, "n layers seq_len hidden_dim"], **kwargs
    ) -> Float[Tensor, "n layers seq_len"]:
        if reps.ndim == 3:
            # We got reps with no seq_len from "get_reps_from_dataset"
            # Add in sequence dimension to broadcast over
            reps = reps.unsqueeze(2)

        b, n_layers, seq_len, hidden_dim = reps.shape
        device = self.device
        probs = torch.zeros((b, n_layers, seq_len), device=device)

        for layer_num in self.config.layers:
            X = reps[:, layer_num, :, :]

            outputs = self.probe[layer_num](X).squeeze(-1)
            assert outputs.shape == (b, seq_len), f"Incorrect outputs shape: {outputs.shape}"
            probs[:, layer_num, :] = torch.sigmoid(outputs)

        return probs.to(torch.float16)

    def predict(
        self,
        reps: Float[Tensor, "b layers seq_len hidden_dim"],
        attention_mask: Optional[Bool[Tensor, "b seq_len"]] = None,
        layer_reduction: str = "mean",
        **kwargs,
    ) -> Float[Tensor, "b"]:

        b, layers, seq_len, hidden_dim = reps.shape
        if b > 1:
            assert attention_mask is not None, "Attention mask must be provided for batch size > 1"

        scores: Float[Tensor, "b layers seq_len"] = self.evaluate(reps)

        # Apply attention mask if provided
        if attention_mask is not None:
            # Expand attention_mask to match scores dimensions
            attention_mask: Float[Tensor, "b layers seq_len"] = attention_mask.unsqueeze(
                1
            ).expand_as(scores)
            assert attention_mask.shape == scores.shape, "Bug in attention mask handling"
            scores = scores.masked_fill(~attention_mask, 0.0)

            # Sum scores and divide by the number of non-masked tokens
            seq_scores = scores.sum(dim=2) / attention_mask.sum(dim=2).clamp(min=1)
        else:
            seq_scores = scores.mean(dim=2)

        # Reduce over layers
        match layer_reduction:
            case "mean":
                final_scores = seq_scores.mean(dim=1)
            case "sum":
                final_scores = seq_scores.sum(dim=1)
            case "max":
                final_scores = seq_scores.max(dim=1).values
            case _:
                raise ValueError(f"Invalid layer_reduction: {layer_reduction}")

        assert final_scores.shape == (b,), f"Expected shape (b,), but got {final_scores.shape}"

        return final_scores

    def load_pretrained(
        self,
        path: Optional[str] = None,
        probe_state: Optional[Any] = None,
    ):
        assert (
            path is not None or probe_state is not None
        ), "Must provide either path or probe_state"
        if path is not None:
            self.probe = torch.load(path)
        else:
            self.probe = probe_state

    def save_probe(self, path: Optional[str] = None) -> None | Any:
        if path is not None:
            torch.save(self.probe, path)
        else:
            return self.probe


class LogisticRegressionMetric(TrainableMetric):
    """Metric that uses logistic regression to classify representations.

    This metric trains a logistic regression model to predict concept labels from model
    representations. The logistic regression model is trained using gradient descent with binary
    cross entropy loss.
    """

    def create_model(self, hidden_size) -> torch.nn.Module:
        return LogisticRegression(hidden_size)


class MLPMetric(TrainableMetric):
    """Metric that uses a multi-layer perceptron (MLP) to classify representations.

    This metric trains a small MLP model to predict concept labels from model representations.
    """

    def create_model(self, hidden_size) -> torch.nn.Module:
        return MLP(hidden_size)


class SAEClassifierMetric(ObfMetric):
    """Metric that uses Sparse Autoencoder (SAE) features to classify representations.

    This metric leverages pre-trained SAEs from the EleutherAI/sae-llama-3-8b-32x model
    to extract interpretable features from model representations. It then trains a simple
    classifier on these features to predict concept labels.

    The metric only works with Llama-3-8b models since it uses SAEs specifically trained
    for that architecture.

    For each layer specified in the config, the metric:
    1. Loads the corresponding pre-trained SAE
    2. Extracts SAE features from the representations
    3. Trains a classifier to predict concept labels from these features

    The final score indicates how well the SAE features can be used to detect the concept.
    """

    def __init__(
        self,
        model: ModelBase,
        data_module: ConceptDataModule,
        config: MetricConfig,
        logger: Logger,
        probe_path: Optional[str] = None,
        pca_path: Optional[str] = None,
    ):
        super().__init__(model, data_module, config, logger, probe_path, pca_path)
        assert (
            "llama" in model.model.config._name_or_path
        ), "SAEClassifierMetric only supports Llama-3-8b models"

    def fit(
        self,
        reps_dataset: RepsDatasetT,
        **kwargs,
    ):
        # Metric should only be used when red_seq_len is 1
        for example in reps_dataset:
            rep = example[0]
            assert rep.shape[2] == 1, "Rep sequence length must be 1 for this metric."

        # Concatenate the reps into one tensor and the labels
        reps = torch.cat([example[0] for example in reps_dataset], dim=0)
        labels = torch.cat([example[1] for example in reps_dataset], dim=0)

        reps = reps.squeeze(dim=2)

        device = self.device
        reps = reps.to(device)
        labels = labels.to(device)

        self.probe: List[SAEClassifier] = []

        for layer in self.config.layers:
            sae = Sae.load_from_hub(
                "EleutherAI/sae-llama-3-8b-32x",
                hookpoint=f"layers.{layer}",
                device=device,
            )
            for param in sae.parameters():
                param.requires_grad = False

            X_train = reps[:, layer, :]
            y_train = labels.to(device)

            dataloader = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(X_train, y_train),
                batch_size=4,  # Hardcoding reasonable value here
                shuffle=True,
            )

            classifier = SAEClassifier(sae).to(device)
            optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3)
            criterion = torch.nn.BCEWithLogitsLoss()

            for X, y in dataloader:
                optimizer.zero_grad()
                outputs = classifier(X).squeeze()
                loss = criterion(outputs, y.float())
                loss.backward()
                optimizer.step()

            self.probe.append(classifier)

    def evaluate(
        self,
        reps: Float[Tensor, "n layers seq_len hidden_dim"],
        **kwargs,
    ) -> Float[Tensor, "n layers"]:

        device = self.device
        reps = reps.to(device)

        if reps.ndim == 3:
            # We got reps with no seq_len from "get_reps_from_dataset"
            # Add in sequence dimension to broadcast over
            reps = reps.unsqueeze(2)

        probs = torch.ones(
            (
                reps.shape[0],
                reps.shape[1],
                reps.shape[2],
            ),
            device=device,
        )

        for idx, layer in enumerate(self.config.layers):
            for seq_pos in range(reps.shape[2]):
                # We add a +1 to layer to account for the input layer (that is 0 in the sae)
                X = reps[:, layer + 1, seq_pos, :]
                outputs = self.probe[idx](X)
                probs[:, layer + 1, seq_pos] = torch.sigmoid(outputs)

        # 1 is "harmful" and 0 is "benign"
        return probs.to(torch.float16)

    def predict(
        self,
        reps: Float[Tensor, "b layers seq_len hidden_dim"],
        attention_mask: Optional[Bool[Tensor, "b seq_len"]] = None,
        layer_reduction: str = "mean",
        **kwargs,
    ) -> Float[Tensor, "b"]:

        scores: Float[Tensor, "b layers seq_len"] = self.evaluate(reps)

        # Select last token
        scores = scores[:, :, -1]

        # {Operator} over layers
        match layer_reduction:
            case "mean":
                scores = scores.mean(dim=1)
            case "sum":
                scores = scores.sum(dim=1)
            case "max":
                scores = scores.max(dim=1).values
            case _:
                raise ValueError(f"Invalid layer_reduction: {layer_reduction}")

        return scores


class GemmaScopeClassifierMetric(ObfMetric):
    """Metric that uses GemmaScope features to detect obfuscation.

    This metric uses pretrained sparse autoencoders (SAEs) from the GemmaScope project to detect
    obfuscation. For each layer specified in config.layers, it loads the corresponding SAE and uses
    it to encode the representations. The metric then looks at specific features (neurons) in the
    SAE's latent space that are known to be relevant for detecting harmful content.

    This metric only works with Gemma-2-2b models.
    """

    def __init__(
        self,
        model: ModelBase,
        data_module: ConceptDataModule,
        config: MetricConfig,
        logger: Logger,
    ):
        self.sae_probes: Dict[int, SAE] = {}
        self.classifiers: Dict[int, LogisticRegression] = {}
        super().__init__(model, data_module, config, logger)

    def fit(
        self,
        reps_dataset: RepsDatasetT,
        **kwargs,
    ):
        device = self.model.device

        # Initialize SAE probes
        for layer_num in self.config.layers:
            sae, _, _ = SAE.from_pretrained(
                release="gemma-scope-2b-pt-res-canonical",
                sae_id=f"layer_{layer_num}/width_16k/canonical",
                device=device,
            )
            for param in sae.parameters():
                param.requires_grad = False
            self.sae_probes[layer_num] = sae

        all_reps = defaultdict(list)
        all_labels = defaultdict(list)

        print("Getting SAE activations")
        for rep, label in reps_dataset:
            for layer_num in self.config.layers:
                hidden_states = rep[:, layer_num + 1, :, :]
                sae = self.sae_probes[layer_num]
                sae_acts = sae.encode_jumprelu(hidden_states.to(device)).squeeze(0)
                all_reps[layer_num].append(sae_acts.to("cpu"))  # Move to cpu to avoid OOM
                all_labels[layer_num].append(label.repeat(sae_acts.shape[0]))

        print("Concatenating SAE activations")
        for layer_num in tqdm.tqdm(self.config.layers):
            all_reps[layer_num] = torch.cat(all_reps[layer_num], dim=0)
            all_labels[layer_num] = torch.cat(all_labels[layer_num], dim=0)

        print(f"Training {len(self.config.layers)} classifiers on layers {self.config.layers}")
        for layer_num in tqdm.tqdm(self.config.layers):
            features = all_reps[layer_num]
            labels = all_labels[layer_num]

            X = features.to(device)
            y = labels.to(device)

            classifier = LogisticRegression(X.shape[1], dtype=X.dtype).to(device)
            optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3)
            criterion = torch.nn.BCEWithLogitsLoss()

            dataloader = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(X, y),
                batch_size=64,
                shuffle=True,
            )

            classifier.train()
            for epoch in range(1):
                for X, y in dataloader:
                    optimizer.zero_grad()
                    outputs = classifier(X).squeeze()
                    loss = criterion(outputs, y)
                    loss.backward()
                    optimizer.step()
                    self.logger.log({f"probe_loss_layer_{layer_num}": loss.item()})

            self.classifiers[layer_num] = classifier

    def evaluate(
        self,
        reps: Float[Tensor, "n layers seq_len hidden_dim"],
        **kwargs,
    ) -> Float[Tensor, "n layers seq_len"]:
        device = reps.device
        if reps.ndim == 3:
            reps = reps.unsqueeze(2)

        b, n_layers, seq_len, hidden_dim = reps.shape
        device = self.device
        probs = torch.zeros((b, n_layers, seq_len), device=device)

        for layer_num in self.config.layers:
            hidden_states = reps[:, layer_num + 1, :, :]
            sae = self.sae_probes[layer_num]
            sae_acts = sae.encode_jumprelu(hidden_states).squeeze()

            outputs = self.classifiers[layer_num](sae_acts).squeeze()
            probs[:, layer_num, :] = torch.sigmoid(outputs)

        return probs.to(torch.float16)

    def predict(
        self,
        reps: Float[Tensor, "b layers seq_len hidden_dim"],
        attention_mask: Optional[Bool[Tensor, "b seq_len"]] = None,
        layer_reduction: str = "mean",
        **kwargs,
    ) -> Float[Tensor, "b"]:
        b, layers, seq_len, hidden_dim = reps.shape
        if b > 1:
            assert attention_mask is not None, "Attention mask must be provided for batch size > 1"

        scores: Float[Tensor, "b layers seq_len"] = self.evaluate(reps)

        # Apply attention mask if provided
        if attention_mask is not None:
            # Expand attention_mask to match scores dimensions
            attention_mask: Float[Tensor, "b layers seq_len"] = attention_mask.unsqueeze(
                1
            ).expand_as(scores)
            assert attention_mask.shape == scores.shape, "Bug in attention mask handling"
            scores = scores.masked_fill(~attention_mask, 0.0)

            # Sum scores and divide by the number of non-masked tokens
            seq_scores = scores.sum(dim=2) / attention_mask.sum(dim=2).clamp(min=1)
        else:
            seq_scores = scores.mean(dim=2)

        match layer_reduction:
            case "mean":
                final_scores = seq_scores.mean(dim=1)
            case "sum":
                final_scores = seq_scores.sum(dim=1)
            case "max":
                final_scores = seq_scores.max(dim=1).values
            case _:
                raise ValueError(f"Invalid layer_reduction: {layer_reduction}")

        assert final_scores.shape == (b,), f"Expected shape (b,), but got {final_scores.shape}"

        return final_scores

    def load_pretrained(self, path: Optional[str] = None, probe_state: Optional[Any] = None):
        pass

    def save_probe(self, path: Optional[str] = None) -> None | Any:
        pass


class GemmaScopeFeatureMetric(ObfMetric):
    def fit(
        self,
        reps_dataset: RepsDatasetT,
        **kwargs,
    ):
        """Select which layers and which features we care about and initialize the SAE."""
        example_rep = reps_dataset[0][0]
        device = example_rep.device

        self.probe: List[SAE] = []

        for layer_num in self.config.layers:
            print(f"Loading SAE for layer {layer_num}")
            sae, _, _ = SAE.from_pretrained(
                release="gemma-scope-2b-pt-res-canonical",
                sae_id=f"layer_{layer_num}/width_16k/canonical",
                device="cuda",
            )
            for param in sae.parameters():
                param.requires_grad = False

            self.probe.append(sae)

    def load_pretrained(self, path: Optional[str] = None, probe_state: Optional[Any] = None):
        pass

    def save_probe(self, path: Optional[str] = None) -> None | Any:
        pass

    def get_feature_dict(self) -> Dict[int, List[int]]:
        """Read the feature list from the config file.

        Config file is a json with keys as layer numbers and values as lists of feature indices.
        """
        assert self.config.feature_list_filepath is not None, "Feature list filepath is None"
        assert os.path.exists(
            self.config.feature_list_filepath
        ), "Feature list file does not exist"

        with open(self.config.feature_list_filepath) as f:
            feature_dict = json.load(f)

        feature_dict = {int(k): [int(x) for x in v] for k, v in feature_dict.items()}
        return feature_dict

    def evaluate(
        self,
        reps: Float[Tensor, "b layers seq_len hidden_dim"],
        **kwargs,
    ) -> Float[Tensor, "b layers seq_len"]:
        """Forward pass through SAE and return feature scores."""
        if reps.ndim == 3:
            # We got reps with no seq_len from "get_reps_from_dataset"
            # Add in sequence dimension to broadcast over
            reps = reps.unsqueeze(2)

        feature_list_dict = self.get_feature_dict()
        max_features = 20  # Max feature_list length

        device = self.device
        reps = reps.to(device)

        probs = torch.zeros(
            (
                reps.shape[0],
                max_features,
                reps.shape[1],
                reps.shape[2],
            ),
            device=device,
        )
        masks = torch.zeros(
            (
                max_features,
                reps.shape[1],
            ),
            device=device,
        )

        for layer_idx, layer_num in enumerate(self.config.layers):
            # We add a +1 to layer to account for the input layer (that is 0 in the sae)
            hidden_states = reps[:, layer_num + 1, :, :]
            sae = self.probe[layer_idx]
            sae_acts: Float[Tensor, "b seq_len d_sae"] = sae.encode_jumprelu(hidden_states)
            features = feature_list_dict[layer_num]
            num_features = len(features)
            sae_acts: Float[Tensor, "b seq_len num_feats"] = sae_acts[:, :, features]
            sae_acts: Float[Tensor, "b num_feats seq_len"] = sae_acts.permute(0, 2, 1)
            probs[:, :num_features, layer_num, :] = sae_acts
            masks[:num_features, layer_num] = 1

        # Apply mask to probes
        masks: Float[Tensor, "1 max_features layers 1"] = masks.unsqueeze(0).unsqueeze(-1)
        masked_probs = probs * masks

        # Calculate masked mean over features
        sum_probs: Float[Tensor, "b layers seq_len"] = masked_probs.sum(dim=1)
        count_features: Float[Tensor, "1 layers 1"] = masks.sum(dim=1)
        masked_mean: Float[Tensor, "b layers seq_len"] = sum_probs / (count_features + 1e-8)

        del probs, masked_probs, sum_probs, count_features

        torch.cuda.empty_cache()

        return masked_mean.to(torch.float16)

    def predict(
        self,
        reps: Float[Tensor, "b layers seq_len hidden_dim"],
        attention_mask: Optional[Bool[Tensor, "b seq_len"]] = None,
        layer_reduction: str = "mean",
        **kwargs,
    ) -> Float[Tensor, "b"]:
        scores: Float[Tensor, "b layers seq_len"] = self.evaluate(reps)

        # {Operator} over layers
        match layer_reduction:
            case "mean":
                scores = scores.mean(dim=1)
            case "sum":
                scores = scores.sum(dim=1)
            case "max":
                scores = scores.max(dim=1).values
            case _:
                raise ValueError(f"Invalid layer_reduction: {layer_reduction}")

        # Return the max score over the seq_len dimension
        return scores.max(dim=-1).values


class MahalanobisDistanceMetric(ObfMetric):
    """OOD detector that measures Mahalanobis distance from benign examples.

    This metric treats benign examples as samples from a multivariate Gaussian distribution
    and measures how far obfuscated examples deviate from this distribution. For each layer,
    it:

    1. Fits a Gaussian distribution to the benign examples by calculating their mean and
       covariance matrix
    2. Measures the Mahalanobis distance between new examples and this distribution

    The Mahalanobis distance accounts for correlations between features and scales each
    dimension appropriately. Higher distances indicate examples that are more anomalous
    compared to the benign distribution.
    """

    def fit(self, reps_dataset: RepsDatasetT, **kwargs):
        # Extract negative examples
        negative_reps_list: List[Float[Tensor, "b layers seq_len hidden_dim"]] = [
            rep for rep, label in reps_dataset if label == 0
        ]

        _, layers, _, hidden_dim = negative_reps_list[0].shape
        # Concatenate negative reps into a single tensor
        # Concat over b and seq_len
        negative_reps = torch.cat(
            [rep.view(-1, rep.shape[1], rep.shape[-1]) for rep in negative_reps_list], dim=0
        )

        # Resulting shape: [total_examples, layers, hidden_dim]
        assert negative_reps.ndim == 3, f"Expected 3 dimensions, but got {negative_reps.ndim}"
        assert (
            negative_reps.shape[1] == layers
        ), f"Expected second dimension to have shape {layers}, but got {negative_reps.shape[1]}"
        assert (
            negative_reps.shape[2] == hidden_dim
        ), f"Expected third dimension to have shape {hidden_dim}, but got {negative_reps.shape[2]}"

        # Calculate mean and covariance for each layer
        self.means = []
        self.inv_covs = []

        # Post refactor, this takes in batches and
        # calculates on the fly.
        for layer in self.config.layers:
            layer_reps = negative_reps[:, layer, :].to(torch.float64)
            mean = torch.mean(layer_reps, dim=0)

            cov = torch.cov(layer_reps.T)
            mean_trace = torch.mean(torch.diag(cov))
            reg = 0.05
            cov = (1 - reg) * cov + reg * mean_trace * torch.eye(cov.shape[0])

            inv_cov = torch.linalg.inv(cov)

            self.means.append(mean)
            self.inv_covs.append(inv_cov)

        self.means = torch.stack(self.means)
        self.inv_covs = torch.stack(self.inv_covs)

    def evaluate(
        self,
        reps: Float[Tensor, "b layers seq_len hidden_dim"],
        **kwargs,
    ) -> Float[Tensor, "b layers seq_len"]:
        b, n_layers, seq_len, hidden_dim = reps.shape
        device = self.device

        # Initialize output tensor
        distances = torch.zeros((b, n_layers, seq_len), device=device)

        for layer_idx, layer in enumerate(self.config.layers):
            layer_reps = reps[:, layer, :, :]
            mean = self.means[layer_idx].to(device)
            inv_cov = self.inv_covs[layer_idx].to(device)

            # Reshape layer_reps to [b * seq_len, hidden_dim]
            layer_reps_flat = layer_reps.reshape(-1, hidden_dim)
            assert layer_reps_flat.shape == (
                b * seq_len,
                hidden_dim,
            ), f"Expected shape [b * seq_len, hidden_dim], but got {layer_reps_flat.shape}"

            # Calculate difference from mean
            diff = layer_reps_flat - mean.unsqueeze(0)

            # Calculate Mahalanobis distance
            # Might be causing problem with numerical stability.
            # Instead can just normalize some other way.
            # Could just divide by hidden dimension.
            mahalanobis_dist = torch.sqrt(torch.sum(torch.matmul(diff, inv_cov) * diff, dim=1))

            assert mahalanobis_dist.shape == (
                b * seq_len,
            ), f"Expected shape {b * seq_len}, but got {mahalanobis_dist.shape}"

            # Reshape back to [b, seq_len]
            mahalanobis_dist = mahalanobis_dist.reshape(b, seq_len)

            # Store the distances
            distances[:, layer, :] = mahalanobis_dist

        assert distances.shape == (
            b,
            n_layers,
            seq_len,
        ), f"Expected shape [b, n_layers, seq_len], but got {distances.shape}"
        return distances.to(torch.float16)

    def predict(
        self,
        reps: Float[Tensor, "b layers seq_len hidden_dim"],
        attention_mask: Optional[Bool[Tensor, "b seq_len"]] = None,
        layer_reduction: str = "mean",
        **kwargs,
    ) -> Float[Tensor, "b"]:

        b, n_layers, seq_len, hidden_dim = reps.shape
        if b > 1:
            assert attention_mask is not None, "Attention mask is required for batch size > 1"

        scores: Float[Tensor, "b layers seq_len"] = self.evaluate(reps)

        # Apply attention mask if provided
        if attention_mask is not None:
            # Expand attention_mask to match scores dimensions
            attention_mask: Float[Tensor, "b layers seq_len"] = attention_mask.unsqueeze(
                1
            ).expand_as(scores)
            assert attention_mask.shape == scores.shape, "Bug in attention mask handling"
            scores = scores.masked_fill(~attention_mask, 0.0)

            # Sum scores and divide by the number of non-masked tokens
            scores = scores.sum(dim=2) / attention_mask.sum(dim=2).clamp(min=1)
        else:
            scores = scores.mean(dim=2)

        # {Operator} over layers
        match layer_reduction:
            case "mean":
                scores = scores.mean(dim=1)
            case "sum":
                scores = scores.sum(dim=1)
            case "max":
                scores = scores.max(dim=1).values
            case _:
                raise ValueError(f"Invalid layer_reduction: {layer_reduction}")

        assert scores.shape == (b,), f"Expected shape (b,), but got {scores.shape}"
        return scores

    def load_pretrained(
        self,
        path: Optional[str] = None,
        probe_state: Optional[Any] = None,
    ):
        assert (
            path is not None or probe_state is not None
        ), "Must provide either path or probe_state"
        if path is not None:
            with open(path, "rb") as f:
                data = pickle.load(f)
        else:
            data = probe_state
        self.means = data["means"]
        self.inv_covs = data["inv_covs"]

    def save_probe(
        self,
        path: Optional[str] = None,
    ) -> None | Any:
        data = {"means": self.means, "inv_covs": self.inv_covs}
        if path is not None:
            with open(path, "wb") as f:
                pickle.dump(data, f)
            return None
        return data


class VAEMetric(ObfMetric):
    """Metric that uses variational autoencoders to detect anomalous representations."""

    def __init__(
        self,
        model: ModelBase,
        data_module: ConceptDataModule,
        config: MetricConfig,
        logger: Logger,
        probe_path: Optional[str] = None,
        pca_path: Optional[str] = None,
    ):
        latent_dim: int = 512  # Size of VAE latent space

        self.latent_dim = latent_dim
        self.vaes: Dict[int, VAE] = {}

        # Learning hyperparams
        self.lr = 1e-4
        self.noise = True
        self.kld_weight = 1.0
        super().__init__(model, data_module, config, logger, probe_path, pca_path)

    def fit(self, reps_dataset: RepsDatasetT, **kwargs):
        # Extract negative examples (benign samples)
        negative_reps_list: List[Float[Tensor, "b layers seq_len hidden_dim"]] = [
            rep for rep, label in reps_dataset if label == 0
        ]

        _, layers, _, hidden_dim = negative_reps_list[0].shape
        # Concatenate negative reps into a single tensor
        negative_reps = torch.cat(
            [rep.view(-1, rep.shape[1], rep.shape[-1]) for rep in negative_reps_list], dim=0
        )
        # Shuffle along axis 0 (examples dimension)
        shuffle_idx = torch.randperm(negative_reps.shape[0])
        negative_reps = negative_reps[shuffle_idx]

        device = self.device

        # Train a VAE for each layer
        for layer in self.config.layers:
            layer_reps = negative_reps[:, layer, :].to(device)

            # Initialize VAE
            vae = VAE(input_dim=hidden_dim, latent_dim=self.latent_dim, dtype=torch.float32).to(
                device
            )

            optimizer = torch.optim.Adam(vae.parameters(), lr=self.lr)
            # Train VAE
            batch_size = 64
            num_epochs = 2
            dataset = torch.utils.data.TensorDataset(layer_reps)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

            print(f"Training VAE for layer {layer}")
            for epoch in tqdm.tqdm(range(num_epochs), desc="Epochs", position=0, leave=True):
                total_loss = 0
                for batch in tqdm.tqdm(dataloader, desc="Batches", position=1, leave=False):
                    optimizer.zero_grad()

                    assert len(batch) == 1
                    x = batch[0].to(torch.float32)

                    assert x.ndim == 2, f"Expected 2D tensor, but got {x.ndim}D tensor"
                    assert x.shape[1] == hidden_dim

                    recon_x, mu, log_var = vae(x, noise=self.noise)
                    assert recon_x.shape == x.shape

                    loss = vae.loss_function(
                        reconstruction=recon_x,
                        input=x,
                        mu=mu,
                        log_var=log_var,
                        kld_weight=self.kld_weight,
                    )["loss"]
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()

                    self.logger.log({f"vae_loss_layer_{layer}": loss.item()})

            vae.eval()
            self.vaes[layer] = vae

    def evaluate(
        self,
        reps: Float[Tensor, "b layers seq_len hidden_dim"],
        **kwargs,
    ) -> Float[Tensor, "b layers seq_len"]:
        b, n_layers, seq_len, hidden_dim = reps.shape
        device = self.device

        reps = reps.to(torch.float32)
        # Initialize output tensor
        scores = torch.zeros((b, n_layers, seq_len), device=device)

        for b_num, single_seq_rep in enumerate(reps):
            assert single_seq_rep.ndim == 3
            for layer in self.config.layers:
                layer_reps = single_seq_rep[layer, :, :].to(device)

                assert layer_reps.ndim == 2
                vae = self.vaes[layer]

                # Calculate reconstruction error
                recon_x, mu, log_var = vae(layer_reps, noise=False)
                loss = vae.loss_function(
                    reconstruction=recon_x,
                    input=layer_reps,
                    mu=mu,
                    log_var=log_var,
                    kld_weight=self.kld_weight,
                    reduce=False,
                )["loss"]

                scores[b_num, layer, :] = loss

        return scores

    def predict(
        self,
        reps: Float[Tensor, "b layers seq_len hidden_dim"],
        attention_mask: Optional[Bool[Tensor, "b seq_len"]] = None,
        layer_reduction: str = "mean",
        **kwargs,
    ) -> Float[Tensor, "b"]:
        b, n_layers, seq_len, hidden_dim = reps.shape
        scores: Float[Tensor, "b layers seq_len"] = self.evaluate(reps)

        if b > 1:
            assert attention_mask is not None, "Attention mask is required for batch evaluation"

        # Apply attention mask if provided
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).expand_as(scores)
            scores = scores.masked_fill(~attention_mask, 0.0)
            scores = scores.sum(dim=2) / attention_mask.sum(dim=2).clamp(min=1)
        else:
            scores = scores.mean(dim=2)

        # Reduce over layers
        num_layers = len(self.config.layers)
        match layer_reduction:
            case "mean":
                scores = scores.sum(dim=1) / num_layers
            case "sum":
                scores = scores.sum(dim=1)
            case "max":
                scores = scores.max(dim=1).values
            case _:
                raise ValueError(f"Invalid layer_reduction: {layer_reduction}")

        assert scores.shape == (b,), f"Expected shape (b,), but got {scores.shape}"
        return scores

    def load_pretrained(
        self,
        path: Optional[str] = None,
        probe_state: Optional[Any] = None,
    ):
        assert (
            path is not None or probe_state is not None
        ), "Must provide either path or probe_state"
        if path is not None:
            with open(path, "rb") as f:
                self.vaes = pickle.load(f)
        else:
            self.vaes = probe_state

    def save_probe(
        self,
        path: Optional[str] = None,
    ) -> None | Any:
        if path is not None:
            with open(path, "wb") as f:
                pickle.dump(self.vaes, f)
            return None
        return self.vaes


# ----- For iterated attack defense ----- #
# Metrics below here are deigned for adversarial training where they add to an ensemble
# each time they are refitted on new obfuscated activations.


class EnsembleMetric(ObfMetric):
    """Base class for an ensemble of metrics that grows during adversarial training.

    This metric maintains a list of individual metrics that grows each time fit() is called.
    During evaluation, it aggregates predictions from all metrics in the ensemble using
    either mean or max aggregation.

    The ensemble is designed for adversarial training scenarios where we want to maintain
    detection capability on both original and obfuscated examples. Each time new obfuscated
    examples are generated, a new metric is trained and added to the ensemble.

    Subclasses must implement create_metric() to specify what type of metric to add to
    the ensemble.

    Note:
        This metric has one additional argument, aggregate_method, which specifies how to
        combine predictions from the ensemble members.
    """

    def __init__(
        self,
        model: ModelBase,
        data_module: ConceptDataModule,
        config: MetricConfig,
        logger: Logger,
        aggregate_method: str = "mean",
        probe_path: Optional[str] = None,
        pca_path: Optional[str] = None,
    ):
        self.metrics: List[ObfMetric] = []
        assert aggregate_method in ["mean", "max"]
        self.aggregate_method = aggregate_method
        super().__init__(model, data_module, config, logger, probe_path, pca_path)

    def fit(
        self,
        reps_dataset: RepsDatasetT,
        **kwargs,
    ):
        new_metric = self.create_metric()
        new_metric.fit(reps_dataset, **kwargs)
        self.metrics.append(new_metric)

    def evaluate(
        self,
        reps: Float[Tensor, "b layers seq_len hidden_dim"],
        **kwargs,
    ) -> Float[Tensor, "b layers seq_len"]:
        results = [metric.evaluate(reps, **kwargs) for metric in self.metrics]
        if self.aggregate_method == "mean":
            return torch.stack(results).mean(dim=0)
        elif self.aggregate_method == "max":
            return torch.stack(results).max(dim=0).values

    def predict(
        self,
        reps: Float[Tensor, "b layers seq_len hidden_dim"],
        attention_mask: Optional[Bool[Tensor, "b seq_len"]] = None,
        layer_reduction: str = "mean",
        **kwargs,
    ) -> Float[Tensor, "b"]:
        results = [
            metric.predict(reps, attention_mask, layer_reduction, **kwargs)
            for metric in self.metrics
        ]
        if self.aggregate_method == "mean":
            return torch.stack(results).mean(dim=0)
        elif self.aggregate_method == "max":
            return torch.stack(results).max(dim=0).values

    def refit(self, attack_defense_epoch: int) -> None:
        new_rep_dataset = self.get_reps_from_dataset(
            self.model,
            self.data_module.train_dataset,  # type: ignore
            self.train_reps_reduce,
            use_tunable_params=True,
        )
        new_metric = self.create_metric()
        new_metric.fit(new_rep_dataset)
        self.metrics.append(new_metric)

    @abstractmethod
    def create_metric(self) -> ObfMetric:
        pass

    def load_pretrained(
        self,
        path: Optional[str] = None,
        probe_state: Optional[Any] = None,
    ):
        assert (
            path is not None or probe_state is not None
        ), "Must provide either path or probe_state"
        if path is not None:
            with open(path, "rb") as f:
                probe_states = pickle.load(f)
            for metric, probe_state in zip(self.metrics, probe_states):
                metric.load_pretrained(probe_state=probe_state)
        else:
            # Upload probe state to each probe individually
            assert len(probe_state) == len(
                self.metrics
            ), "Number of probe states must match number of metrics"
            for metric, probe_state in zip(self.metrics, probe_state):
                metric.load_pretrained(probe_state=probe_state)

    def save_probe(
        self,
        path: Optional[str] = None,
    ) -> None | Any:

        if path is not None:
            probe_states = [metric.save_probe() for metric in self.metrics]
            with open(path, "wb") as f:
                pickle.dump(probe_states, f)
            return None
        else:
            return [metric.save_probe() for metric in self.metrics]


class MeanEnsembleMetric(EnsembleMetric):
    """Ensemble metric that averages predictions from multiple base metrics."""

    def __init__(
        self,
        model: ModelBase,
        data_module: ConceptDataModule,
        config: MetricConfig,
        logger: Logger,
        probe_path: Optional[str] = None,
        pca_path: Optional[str] = None,
    ):
        super().__init__(
            model=model,
            data_module=data_module,
            config=config,
            logger=logger,
            probe_path=probe_path,
            pca_path=pca_path,
            aggregate_method="mean",
        )


class MaxEnsembleMetric(EnsembleMetric):
    """Ensemble metric that takes the maximum prediction from multiple base metrics."""

    def __init__(
        self,
        model: ModelBase,
        data_module: ConceptDataModule,
        config: MetricConfig,
        logger: Logger,
        probe_path: Optional[str] = None,
        pca_path: Optional[str] = None,
    ):
        super().__init__(
            model=model,
            data_module=data_module,
            config=config,
            logger=logger,
            probe_path=probe_path,
            pca_path=pca_path,
            aggregate_method="max",
        )


class MeanEnsembleMLPMetric(MeanEnsembleMetric):
    """Ensemble metric that averages predictions from multiple MLP metrics."""

    def create_metric(self) -> ObfMetric:
        return MLPMetric(
            self.model,
            self.data_module,
            self.config,
            self.logger,
        )


class MeanEnsembleMeanDiffCosineSimMetric(MeanEnsembleMetric):
    """Ensemble metric that averages predictions from multiple MeanDiffCosineSim metrics."""

    def create_metric(self) -> ObfMetric:
        return MeanDiffCosineSimMetric(
            self.model,
            self.data_module,
            self.config,
            self.logger,
        )


class MeanEnsembleLogisticRegressionMetric(MeanEnsembleMetric):
    """Ensemble metric that averages predictions from multiple logistic regression metrics."""

    def create_metric(self) -> ObfMetric:
        return LogisticRegressionMetric(
            self.model,
            self.data_module,
            self.config,
            self.logger,
        )


class MaxEnsembleMLPMetric(MaxEnsembleMetric):
    """Ensemble metric that takes maximum predictions from multiple MLP metrics."""

    def create_metric(self) -> ObfMetric:
        return MLPMetric(
            self.model,
            self.data_module,
            self.config,
            self.logger,
        )


class MaxEnsembleMeanDiffCosineSimMetric(MaxEnsembleMetric):
    """Ensemble metric that takes maximum predictions from multiple MeanDiffCosineSim metrics."""

    def create_metric(self) -> ObfMetric:
        return MeanDiffCosineSimMetric(
            self.model,
            self.data_module,
            self.config,
            self.logger,
        )


class MaxEnsembleLogisticRegressionMetric(MaxEnsembleMetric):
    """Ensemble metric that takes maximum predictions from multiple logistic regression metrics."""

    def create_metric(self) -> ObfMetric:
        return LogisticRegressionMetric(
            self.model,
            self.data_module,
            self.config,
            self.logger,
        )
