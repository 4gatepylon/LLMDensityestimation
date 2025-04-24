from __future__ import annotations
from pathlib import Path
from typing import List
import numpy as np
from sklearn.decomposition import IncrementalPCA as CPU_IPCA
from tqdm import tqdm
import gc
import click
import joblib

def incremental_pca(files: List[str], mu: np.ndarray, k: int, limit: int | None = None) -> CPU_IPCA:
    """
    Perform Incremental PCA on activation data stored in multiple files,
    potentially using only a subset of the data.

    Args:
        files: List of paths to .npy files containing activation batches.
        mu: The pre-calculated mean vector of the activations.
        k: The number of principal components to compute.
        limit: Maximum number of activation vectors to consider. If None, use all.

    Returns:
        The fitted IncrementalPCA object.
    """
    ipca = CPU_IPCA(n_components=k)
    mu32 = mu.astype(np.float32)
    processed_count = 0

    print(f"Running IPCA for {k} components on {limit if limit else 'all'} tokens...")
    bar = tqdm(total=limit if limit else None, desc="IPCA", unit="tok")

    for f in files:
        if limit is not None and processed_count >= limit:
            break

        X = np.load(f).astype(np.float32)
        take = len(X)
        if limit is not None:
            remaining_limit = limit - processed_count
            take = min(take, remaining_limit)

        if take <= 0:
            continue

        X_chunk = X[:take] - mu32
        norm_mask = np.linalg.norm(X_chunk, axis=1) > 1e-6
        X_filtered = X_chunk[norm_mask]

        if X_filtered.shape[0] > 0:
            ipca.partial_fit(X_filtered)
            update_count = X_chunk.shape[0] # Update bar based on original chunk size before filtering
            bar.update(update_count)
            processed_count += update_count

        # Clean up
        del X, X_chunk, X_filtered, norm_mask
        gc.collect()

    bar.close()
    print(f"IPCA finished after processing {processed_count:,} tokens.")
    return ipca

def project_onto_ipca(ipca: CPU_IPCA, X: np.ndarray, k: int) -> np.ndarray:
    if k > ipca.n_components_:
        raise ValueError(f"k ({k}) must be <= ipca.n_components_ ({ipca.n_components_})")
    
    # Transform the data using the IPCA model
    X_centered = X - ipca.mean_
    X_transformed = ipca.transform(X_centered)
    
    # Return only the first k components
    return X_transformed[:, :k]

def project_files_onto_ipca(files_folder: Path, ipca: CPU_IPCA, k: int, output_folder: Path) -> np.ndarray:
    assert not output_folder.exists(), f"Output folder {output_folder} already exists"
    output_folder.mkdir(parents=True, exist_ok=True)

    np_files = list(files_folder.glob("*.npy"))
    for f in tqdm(np_files, desc="Projecting files"):
        rel_path = f.relative_to(files_folder)
        out_path = output_folder / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # 1. Load file
        X = np.load(f).astype(np.float32)
        
        # 2. Project onto the first k components (make sure to subtract mean)
        X_proj = project_onto_ipca(ipca, X, k)
        
        # 3. Save the projected data
        np.save(out_path, X_proj)
        
        # Clean up
        del X, X_proj
        gc.collect()

@click.command()
@click.option("--files_folder", "-f", "-input", "-i", type=click.Path(exists=True))
@click.option("--output_folder", "-o", "-out", "-output", type=click.Path())
@click.option("--k", "-k", type=int)
def main(files_folder: Path, output_folder: Path, k: int):
    files_folder = Path(files_folder)
    output_folder = Path(output_folder)
    # ipca = incremental_pca(files_folder, output_folder, k) # Turns out no need since it's stored
    ipca = joblib.load(files_folder / "pca.joblib")
    assert isinstance(ipca, CPU_IPCA), f"IPCA model is not a CPU_IPCA: {type(ipca)}"
    project_files_onto_ipca(files_folder, ipca, k, output_folder)

if __name__ == "__main__":
    main()
