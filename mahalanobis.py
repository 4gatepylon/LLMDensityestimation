from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
import einops
import math
from jaxtyping import Float
import matplotlib.pyplot as plt
from typing import Tuple, Optional, List
from transformers import AutoTokenizer, AutoModel
import numpy as np

"""
Implement vectorized and fast GMM training for single-gaussian + helper plotting functions
(and other general helper stuff).

(i.e. mahalanobis distance centroid optimizer... it's in `anomaly_march_pre26`)
"""


class Mahalanobis:
    @staticmethod
    def gaussian_params(
        x: Float[torch.Tensor, "batch data layer dim"],
    ) -> Tuple[
        Float[torch.Tensor, "batch layer dim"],
        Float[torch.Tensor, "batch layer dim dim"],
    ]:
        """
        Calculate the parameters for an MLE gaussian (single gaussian) to fit your data.
        """
        assert x.ndim == 4
        # Get the means
        means = einops.reduce(
            x, "batch data layer dim -> batch layer dim", reduction="mean"
        )
        assert means.shape == (x.shape[0], x.shape[2], x.shape[3]), f"means.shape: {means.shape} != (x.shape[0], x.shape[1], x.shape[2]): {x.shape}" # fmt: skip
        data_centered = x - einops.rearrange(
            means, "batch layer dim -> batch 1 layer dim"
        )
        # X.T @ X where X is tall
        sigma = (
            einops.einsum(
                data_centered,
                data_centered,
                "batch data layer dim1, batch data layer dim2 -> batch layer dim1 dim2",
            )
            / x.shape[1]
        )
        assert sigma.shape == (x.shape[0], x.shape[2], x.shape[3], x.shape[3]), f"sigma.shape: {sigma.shape} != (x.shape[0], x.shape[1], x.shape[2], x.shape[2]): {x.shape}" # fmt: skip
        return means, sigma

    @staticmethod
    def get_scores(
        x: Float[torch.Tensor, "batch data layer dim"],
        means: Float[torch.Tensor, "batch layer dim"],
        sigmas: Float[torch.Tensor, "batch layer dim dim"],
        sigmas_inv: Float[torch.Tensor, "batch layer dim dim dim"],
        do_full_logprob: bool = False,
    ) -> Float[torch.Tensor, "batch data layer"]:
        """
        For multiple datasets at once, calculate the scores.

        Score is basically logprob, but you have the option to exclude the normalizing
        constant (which depends on your gaussian's covariance matrix).
        """
        assert x.ndim == 4
        assert means.ndim == 3 and sigmas_inv.ndim == 4
        batch, data, layer, dim = x.shape
        assert means.shape == (batch, layer, dim), f"means.shape: {means.shape} != (batch, layer, dim): {(batch, layer, dim)}" # fmt: skip
        assert sigmas.shape == (batch, layer, dim, dim), f"sigmas.shape: {sigmas.shape} != (batch, layer, dim, dim): {(batch, layer, dim, dim)}" # fmt: skip
        assert sigmas_inv.shape == (batch, layer, dim, dim), f"sigmas_inv.shape: {sigmas_inv.shape} != (batch, layer, dim, dim): {(batch, layer, dim, dim)}" # fmt: skip
        x_centered = x - einops.rearrange(
            means, "batch layer dim -> batch 1 layer dim", dim=x.shape[-1]
        )
        _left = einops.einsum(x_centered, sigmas_inv, "batch data layer dim1, batch layer dim1 dim2 -> batch data layer dim2")# , dim1=x.shape[-1], dim2=x.shape[-1]) # left multiply vector; fmt: skip
        _left_and_right = einops.einsum(_left, x_centered, "batch data layer dim2, batch data layer dim2 -> batch data layer")# , dim1=x.shape[-1], dim2=x.shape[-1]) # right multiply vector; fmt: skip
        _exp = -0.5 * _left_and_right
        assert _exp.shape == (batch, data, layer), f"_exp.shape: {_exp.shape} != (batch, data, layer): {(batch, data, layer)}" # fmt: skip
        # PDF: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
        # NOTE this seems faster for me???
        # In [39]: %timeit U, S, Vh = torch.linalg.svd(A, full_matrices=False)
        # %timeit 1.26 s ± 70.9 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

        # In [40]: %timeit torch.linalg.eigvals(A)
        # 5.71 s ± 485 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
        # NOTE: this calculation is technically not needed since it does not depend on x
        if do_full_logprob:
            _, S, _ = torch.linalg.eigvals(sigmas, full_matrices=True)
            assert S.ndim == 4  # batch layer dim dim
            S_log = -0.5 * torch.log(S).sum(dim=-1).sum(dim=-1)
            assert S_log.ndim == 2  # batch layer
            S_log = einops.rearrange(S_log, "batch layer -> batch 1 layer")
            pi_log = (-x.shape[-1] * 0.5) * torch.log(
                2 * torch.tensor(math.pi)
            ).reshape(1, 1, 1)
            assert pi_log.ndim == 0
            pi_log = einops.rearrange(pi_log, "-> 1 1")
            log_prob = _exp + S_log + pi_log
            return log_prob
        else:
            return _exp


def plot_score_distributions(
    scores_dict,
    dim=None,
    bins=50,
    alpha=0.7,
    figsize_individual=(15, 5),
    figsize_combined=(10, 6),
    layer: Optional[str] = None,
):
    """
    NOTE: copied from `anomaly_march_pre26`: Create a figure with subplots for individual distributions
    """
    # Process data based on dimension parameter
    processed_scores = {}
    for i, (name, scores) in enumerate(scores_dict.items()):
        if dim is None:
            processed_scores[name] = scores.flatten()
        elif isinstance(dim, int):
            processed_scores[name] = scores[:, dim].flatten()
        else:  # dim is a list or tuple
            # Select only the specified dimensions
            processed_scores[name] = scores[:, dim].flatten()

    # Plot all distributions on the same plot for comparison
    plt.figure(figsize=figsize_combined)
    for name, scores in processed_scores.items():
        plt.hist(scores, bins=bins, alpha=alpha / 2, label=name)

    dim_str = (
        (f"Layer at Dimension(s): {dim}" if dim is not None else "All Layers")
        if layer is None
        else f"Layer: {layer}"
    )
    plt.title(f"Score Distributions Comparison ({dim_str})")
    plt.xlabel("Score")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.show()


class MahalanobisLikelihoodScore(nn.Module):
    """
    Takes in a batch of activations and returns something that is equal to
    the negative logprob, scaled and shifted by some constants (w.r.t. inputs
    that vary).
    """

    def __init__(self, inv_cov: Float[Tensor, "d d"], mean: Float[Tensor, "d"]):
        super().__init__()
        assert inv_cov.ndim == 2
        assert mean.ndim == 1
        self.inv_cov = nn.Parameter(inv_cov)
        self.mean = nn.Parameter(mean)

    def forward(self, x: Float[Tensor, "b d"]) -> Float[Tensor, "b"]:
        assert x.ndim == 2, f"x.shape: {x.shape} has more than 2D"
        c = x - einops.rearrange(self.mean, "d -> 1 d")
        _right = einops.einsum(self.inv_cov, c, "d1 d2, b d2 -> b d1")
        _both = einops.einsum(c, _right, "b d1, b d1 -> b")
        return -0.5 * _both  # This is what you want to MAXIMIZE


class TextAnomalyDetector:
    """
    Take in a model name and use it to get the embeddings of the text.

    NOTE: this is Claude-generated AI SLOP that is probably correct.
    """

    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.means = None
        self.sigmas = None
        self.sigmas_inv = None

    def get_embeddings(self, texts: List[str], batch_size: int = 32) -> torch.Tensor:
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            encoded = self.tokenizer(
                batch_texts, padding=True, truncation=True, return_tensors="pt"
            )
            encoded = {k: v.to(self.device) for k, v in encoded.items()}

            with torch.no_grad():
                outputs = self.model(**encoded)
                # Use CLS token as sentence embedding
                embeddings = outputs.last_hidden_state[:, 0, :]

            all_embeddings.append(embeddings.cpu())

        return torch.cat(all_embeddings, dim=0)

    def fit(self, texts: List[str], batch_size: int = 32) -> MahalanobisLikelihoodScore:
        # Get embeddings
        embeddings = self.get_embeddings(texts, batch_size)

        # Reshape for Mahalanobis calculations (add batch and layer dimensions)
        # [num_samples, embedding_dim] -> [1, num_samples, 1, embedding_dim]
        x = embeddings.unsqueeze(0).unsqueeze(2)

        # Calculate Gaussian parameters
        self.means, self.sigmas = Mahalanobis.gaussian_params(x)

        # Calculate inverse of covariance matrices
        # Add small value to diagonal for numerical stability
        eps = 1e-6
        dim = self.sigmas.shape[-1]
        self.sigmas = self.sigmas + torch.eye(dim).unsqueeze(0).unsqueeze(0) * eps
        self.sigmas_inv = torch.linalg.inv(self.sigmas)

        return MahalanobisLikelihoodScore(self.sigmas_inv, self.means)

    def score_samples(
        self, texts: List[str], batch_size: int = 32, full_logprob: bool = False
    ) -> torch.Tensor:
        # Get embeddings for new texts
        embeddings = self.get_embeddings(texts, batch_size)

        # Reshape for Mahalanobis calculations
        x = embeddings.unsqueeze(0).unsqueeze(2)

        # Calculate anomaly scores
        scores = Mahalanobis.get_scores(
            x, self.means, self.sigmas, self.sigmas_inv, do_full_logprob=full_logprob
        )

        # Flatten the layer dimension and return
        return scores.squeeze(0).squeeze(1)

    def predict_anomalies(
        self,
        texts: List[str],
        threshold: Optional[float] = None,
        contamination: float = 0.05,
        batch_size: int = 32,
    ) -> Tuple[np.ndarray, np.ndarray]:
        scores = self.score_samples(texts, batch_size).numpy()

        if threshold is None:
            # Set threshold based on contamination parameter
            threshold = np.percentile(scores, (1 - contamination) * 100)

        # Lower scores (more negative) = more anomalous in Mahalanobis distance
        predictions = np.where(
            scores < threshold, -1, 1
        )  # -1 for anomaly, 1 for normal

        return predictions, scores
