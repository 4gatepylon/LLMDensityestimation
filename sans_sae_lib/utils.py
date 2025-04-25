from __future__ import annotations

from typing import Optional
import numpy as np
import torch
from jaxtyping import Float
import matplotlib.pyplot as plt
import tqdm
import itertools
import shutil
from pathlib import Path

def compute_cosine_similarity(
        vectors1: Float[torch.Tensor, "batch d_model"],
        vectors2: Float[torch.Tensor, "batch d_model"]
) -> Float[torch.Tensor, "batch batch"]:
    """
    By Claude. Compute the cosine similarity between two sets of vectors
    (basically this is what you might call a "kernel" especially if
    the two sets of vectors are the same ones, but one is transposed).
    """
    # Normalize vectors
    vectors1_norm = vectors1 / torch.norm(vectors1, dim=0, keepdim=True)
    vectors2_norm = vectors2 / torch.norm(vectors2, dim=0, keepdim=True)
    
    # Compute cosine similarity matrix
    return torch.matmul(vectors1_norm.T, vectors2_norm)

def plot_heatmap(
        matrix: Float[torch.Tensor, "batch batch"],
        title: str,
        xlabel: str,
        ylabel: str,
        figsize: tuple[int, int] = (14, 12),
        cmap: str = 'viridis',
        vmin: float = -1,
        vmax: float = 1,
        save_to_file: str | None = None
) -> Float[torch.Tensor, "batch batch"]:
    """
    By Claude. Plot a heatmap from a matrix of numbers.
    """
    plt.figure(figsize=figsize)
    
    # Convert to numpy if it's a tensor
    if isinstance(matrix, torch.Tensor):
        matrix = matrix.cpu().numpy()
    
    # Use imshow instead of seaborn's heatmap for better performance with large matrices
    plt.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
    plt.colorbar(label='Cosine Similarity')
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    # Force the plot to be drawn
    plt.draw()
    
    
    # Save if requested
    # TODO(Adriano) figure out how to be able to BOTH save and show the plot
    if save_to_file:
        plt.savefig(save_to_file)
    else:
        plt.show()
    
    # Close the plot after everything is done
    plt.close()

def plot_cosine_kernel(
        eigenvecs_ins: Float[torch.Tensor, "batch d_model"],
        eigenvecs_outs: Float[torch.Tensor, "batch d_model"],
        save_to_file: str | None = None,
        vecs_title1: str = 'SAE Outs Eigenvectors',
        vecs_title2: str = 'SAE Ins Eigenvectors',
        force_positive: bool = False
) -> Float[torch.Tensor, "batch batch"]:
    """
    By Claude. Plot for two pairs of vectors the similarity between the
    pairs in the cartesian product of the two sets of vectors.
    """
    # Compute cosine similarity table
    cosine_sim = compute_cosine_similarity(eigenvecs_ins, eigenvecs_outs)
    if force_positive:
        # Negative cosine value means that it's matching the "opposite" in this interpretation
        cosine_sim = cosine_sim.abs() 
    
    # Plot the cosine similarity matrix
    plot_heatmap(
        cosine_sim, 
        f'Cosine Similarity (Kernel) Between {vecs_title1} and {vecs_title2}',
        vecs_title1,
        vecs_title2,
        save_to_file=save_to_file
    )
    
    return cosine_sim


def plot_pca_histogram(
        activations_flat: Float[torch.Tensor, "batch d_model"],
        # Parameters to take the PCA
        mean: Float[torch.Tensor, "d_model"],
        eigenvectors: Float[torch.Tensor, "d_model d_model"],
        # Indices of PCA components to plot (0 is highest variance)
        i: int,
        j: int,
        bins: int = 200,
        title: str | None = None,
        save_to_file: str | None = None,
        reversed: bool = True,
        accumulation_values: Optional[Float[torch.Tensor, "batch"]] = None,
        normalize_accumulation_values_by_n_in_bin: bool = False,
        normalize_accumulation_values_by_n_total: bool = False
) -> None:
    """
    Project SAE activations onto PCA components i and j and create a 2D histogram.

    Technically you could use it to see the histogram under any projection.
    
    Args:
        activations_flat: Tensor of shape [batch, dim]
        mean: Mean vector of shape [dim]
        eigenvectors: Matrix of eigenvectors of shape [dim, dim]
        i, j: Indices of PCA components to plot (0 is highest variance)
        bins: Number of histogram bins
        title: Optional title for the plot
        save_to_file: Optional path to save the plot
        reversed: Whether to reverse the indices of the PCA components
        accumulation_values: Optional tensor of shape [batch] to accumulate the values
            (if provided, then the instead of outputting a histogram, this will output
            a plot if the values accumulated in those bins)
        normalize_accumulation_values_by_n_in_bin: Whether to normalize the accumulation values
            by the number of values in the bin; IGNORED IF `accumulation_values` IS NOT PROVIDED
        normalize_accumulation_values_by_n_total: Whether to normalize the accumulation values
            by the total number of values; IGNORED IF `accumulation_values` IS NOT PROVIDED
    """

    # Convert to numpy if needed
    if isinstance(activations_flat, torch.Tensor):
        activations_flat = activations_flat.cpu().numpy()
    if isinstance(mean, torch.Tensor):
        mean = mean.cpu().numpy()
    if isinstance(eigenvectors, torch.Tensor):
        eigenvectors = eigenvectors.cpu().numpy()
    if accumulation_values is not None:
        if normalize_accumulation_values_by_n_in_bin and normalize_accumulation_values_by_n_total:
            raise ValueError("Cannot normalize by both n_in_bin and n_total")
    
    # Get the PCA components (note: eigenvectors are in descending order of variance)
    # So we need to reverse the indices
    i = eigenvectors.shape[1] - 1 - i if reversed else i
    j = eigenvectors.shape[1] - 1 - j if reversed else j
    W = np.stack([eigenvectors[:, i], eigenvectors[:, j]])
    
    # Center the data
    X = activations_flat - mean
    
    # Project onto the PCA components
    P = X @ W.T
    
    # Create histogram bins
    x_min, x_max = P[:, 0].min(), P[:, 0].max()
    y_min, y_max = P[:, 1].min(), P[:, 1].max()
    x_bins = np.linspace(x_min, x_max, bins + 1)
    y_bins = np.linspace(y_min, y_max, bins + 1)
    
    # Create 2D histogram
    H = None
    if accumulation_values is None:
        H, _, _ = np.histogram2d(P[:, 0], P[:, 1], bins=(x_bins, y_bins))
    else:
        # https://numpy.org/doc/2.2/reference/generated/numpy.histogram.html
        # Convert accumulation values to numpy if needed
        if isinstance(accumulation_values, torch.Tensor):
            accumulation_values = accumulation_values.cpu().numpy()
            
        # Use histogram2d with weights
        H, _, _ = np.histogram2d(P[:, 0], P[:, 1], bins=(x_bins, y_bins), weights=accumulation_values)
        
        if normalize_accumulation_values_by_n_in_bin:
            # Get counts for normalization
            counts, _, _ = np.histogram2d(P[:, 0], P[:, 1], bins=(x_bins, y_bins))
            mask = counts > 0
            H[mask] /= counts[mask]
        elif normalize_accumulation_values_by_n_total:
            H /= len(accumulation_values)
    assert H is not None
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.imshow(np.log1p(H).T, origin="lower", aspect="auto",
               extent=[x_min, x_max, y_min, y_max],
               cmap="viridis")
    plt.colorbar(label="log(count)")
    plt.xlabel(f"PC{i+1}")
    plt.ylabel(f"PC{j+1}")
    if title:
        plt.title(title)
    plt.tight_layout()
    
    if save_to_file:
        plt.savefig(save_to_file, dpi=300)
    else:
        plt.show()
    plt.close()


def plot_all_nc2_top_pcs(
        # Projection parameters MUST be provided...
        n: int,
        activations: Float[torch.Tensor, "batch d_model"],
        mean: Float[torch.Tensor, "d_model"],
        eigenvectors: Float[torch.Tensor, "d_model d_model"],
        output_folder: Path,
        # ...but other parameters are optional
        **kwargs
) -> None:
    """
    Plot all combinations of PCs.

    Will NOT clobber existing files, but it will clobber an empty folder
    (since that should be meaningless).
    """
    if output_folder.exists() and len(list(output_folder.glob("*"))) == 0:
        shutil.rmtree(output_folder)
    output_folder.mkdir(parents=True, exist_ok=False)
    for i, j in tqdm.tqdm(itertools.combinations(range(n), 2), desc="Plotting PCA histograms", total=n * (n - 1) // 2):
        plot_pca_histogram(
            activations,
            mean,
            eigenvectors,
            i, j,
            **kwargs,
            save_to_file=output_folder / f"pc{i+1}_pc{j+1}.png"
        )

def plot_all_nc2_top_pcs_errs(
        # ...
        n_pcs: int,
        # Projection (binning) parameters
        activations: Float[torch.Tensor, "layer batch d_model"],
        mean: Float[torch.Tensor, "layer d_model"],
        eigenvectors: Float[torch.Tensor, "layer d_model d_model"],
        # Error parameters
        err_array: Float[torch.Tensor, "layer batch"],
        normalize_accumulation_values_by_n_in_bin: bool,
        normalize_accumulation_values_by_n_total: bool,
        # Output parameters
        output_folder: Path,
        # etc...
        **kwargs
) -> None:
    """
    Plot all combinations of PCs.
    """
    if output_folder.exists() and len(list(output_folder.glob("*"))) == 0:
        shutil.rmtree(output_folder)
    output_folder.mkdir(parents=True, exist_ok=False)
    for i, j in tqdm.tqdm(itertools.combinations(range(n_pcs), 2), desc="Plotting PCA histograms", total=n_pcs * (n_pcs - 1) // 2):
        plot_pca_histogram(
            activations,
            mean,
            eigenvectors,
            i, j,
            **kwargs,
            save_to_file=output_folder / f"pc{i+1}_pc{j+1}.png",
            accumulation_values=err_array, # batch
            normalize_accumulation_values_by_n_in_bin=normalize_accumulation_values_by_n_in_bin,
            normalize_accumulation_values_by_n_total=normalize_accumulation_values_by_n_total
        )


def delete_folder_if_empty(folder: Path) -> None:
    """
    Delete a folder if it is empty.
    """
    if folder.exists() and len(list(folder.glob("*"))) == 0:
        shutil.rmtree(folder)
