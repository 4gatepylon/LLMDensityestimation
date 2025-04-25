from __future__ import annotations
import pydantic
import torch
from jaxtyping import Float, Int
from typing import List, Tuple
import einops

# TODO(Adriano) find a more scalable solution than this plz
class FlattenedExtractedActivations(pydantic.BaseModel):
    """
    A container for the data that we will want to visualize and
    analyze generally.
    """
    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    # Data
    sae_ins: Float[torch.Tensor, "layer batchseq d_model"]
    sae_outs_per_k: Float[torch.Tensor, "ks layer batchseq d_model"]
    ln2s: Float[torch.Tensor, "layer batchseq d_model"]
    ln2s_saed_per_k: Float[torch.Tensor, "ks layer batchseq d_model"]

    # Same but centered
    sae_ins_centered: Float[torch.Tensor, "layer batchseq d_model"]
    sae_outs_per_k_centered: Float[torch.Tensor, "ks layer batchseq d_model"]
    ln2s_centered: Float[torch.Tensor, "layer batchseq d_model"]
    ln2s_saed_per_k_centered: Float[torch.Tensor, "ks layer batchseq d_model"]

    # Means help us analyze, note that we will use keepdim=True
    sae_ins_means: Float[torch.Tensor, "layer 1 d_model"]
    sae_outs_per_k_means: Float[torch.Tensor, "ks layer 1 d_model"]
    ln2s_means: Float[torch.Tensor, "layer 1 d_model"]
    ln2s_saed_per_k_means: Float[torch.Tensor, "ks layer 1 d_model"]

    # This is basically what we are going to analyze/visualize (look above)
    # Errors etc... for the residual stream (impacts of the SAE)
    res_sae_var_explained: Float[torch.Tensor, "ks layer batchseq"]
    res_sae_mse: Float[torch.Tensor, "ks layer batchseq"]
    res_sae_error_norms: Float[torch.Tensor, "ks layer batchseq"]

    # Same but for the ln2
    ln2_sae_var_explained: Float[torch.Tensor, "ks layer batchseq"]
    ln2_sae_mse: Float[torch.Tensor, "ks layer batchseq"]
    ln2_sae_error_norms: Float[torch.Tensor, "ks layer batchseq"]

    # Also this below... we will do PCA and then this is where we get this from
    # NOTE you can get the explained variance per component from the eigenvalues
    # (that one divided by the sum)
    #
    # Res with nothing applied
    sae_ins_pca_eigenvectors: Float[torch.Tensor, "layer d_model d_model"]
    sae_ins_pca_eigenvalues: Float[torch.Tensor, "layer d_model"]
    # Res with sae applied
    sae_outs_per_k_pca_eigenvectors: Float[torch.Tensor, "ks layer d_model d_model"]
    sae_outs_per_k_pca_eigenvalues: Float[torch.Tensor, "ks layer d_model"]
    # ln2 with nothing applied
    ln2s_pca_eigenvectors: Float[torch.Tensor, "layer d_model d_model"]
    ln2s_pca_eigenvalues: Float[torch.Tensor, "layer d_model"]
    # ln2 with sae applied
    ln2s_saed_per_k_pca_eigenvectors: Float[torch.Tensor, "ks layer d_model d_model"]
    ln2s_saed_per_k_pca_eigenvalues: Float[torch.Tensor, "ks layer d_model"]
    

class ExtractedActivations(pydantic.BaseModel):
    """
    A container for the outputs of `extract_activations` from the
    `ResidAndLn2Comparer` class. It can be converted into a `FlattenedExtractedActivations`
    to get data you can analyze and visualize.
    """
    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)
    sae_ins: Float[torch.Tensor, "layer batch seq d_model"]
    sae_outs_per_k: Float[torch.Tensor, "ks layer batch seq d_model"]
    ln2s: Float[torch.Tensor, "layer batch seq d_model"]
    ln2s_saed_per_k: Float[torch.Tensor, "ks layer batch seq d_model"]

    def calculate_error_data(
            self,
            datasets: List[torch.Tensor, "layer batchseq d_model"] | List[torch.Tensor, "ks layer batchseq d_model"]
        ) -> Tuple[
            List[Float[torch.Tensor, "ks layer batchseq"]],
            List[Float[torch.Tensor, "ks layer batchseq"]],
            List[Float[torch.Tensor, "ks layer batchseq"]],
    ]:
        """
        Return
        ```
        res_sae_var_explained: Float[torch.Tensor, "layer batchseq"]
        res_sae_mse: Float[torch.Tensor, "layer batchseq"]
        res_sae_error_norms: Float[torch.Tensor, "layer batchseq"]
        ...
        same stuff for ln2
        ```
        """
        assert len(datasets) == 4, f"len(datasets)={len(datasets)} must be 4"
        assert datasets[0].shape == datasets[2].shape, f"datasets[0].shape={datasets[0].shape} must be datasets[2].shape={datasets[2].shape}" # fmt: skip
        assert datasets[1].shape == datasets[3].shape, f"datasets[1].shape={datasets[1].shape} must be datasets[3].shape={datasets[3].shape}" # fmt: skip
        assert datasets[0].ndim == 3, f"datasets[0].ndim={datasets[0].shape} ndim must be 3"
        assert datasets[1].ndim == 4, f"datasets[1].ndim={datasets[1].shape} ndim must be 4"
        assert datasets[0].shape == datasets[1].shape[1:], f"datasets[0].shape={datasets[0].shape} must be datasets[1].shape[1:]=={datasets[1].shape[1:]}" # fmt: skip
        data_pairs = [
            # NOTE 2nd elem of tuple is per-k and 1st is NOT per-k
            # saes_in, saes_out
            (datasets[0], datasets[1]),
            # ln2s, ln2s_saed
            (datasets[2], datasets[3]),
        ]
        batchseq_dim_sae_in = 1
        layer, batchseq, d_model = datasets[0].shape
        kidxs = datasets[1].shape[0]
        vars_explained = []
        mses = []
        error_norms = []
        for sae_in, sae_out in data_pairs:
            # Calculate
            sq_errs = (sae_in.unsqueeze(0) - sae_out).pow(2).sum(dim=-1) # norms of errors
            assert sq_errs.shape == (kidxs, layer, batchseq), f"sq_errs.shape={sq_errs.shape} must be (kidxs, layer, batchseq)={kidxs, layer, batchseq}" # fmt: skip
            var = (sae_in - sae_in.mean(dim=batchseq_dim_sae_in, keepdim=True)).pow(2).sum(dim=-1).unsqueeze(0) # variance of 1st dim then sum those
            assert var.shape == (1, layer, batchseq), f"var.shape={var.shape} must be (1, layer, batchseq)={1, layer, batchseq}" # fmt: skip
            explained_variances = 1 - sq_errs / var
            assert explained_variances.shape == (kidxs, layer, batchseq), f"explained_variances.shape={explained_variances.shape} must be (kidxs, layer, batchseq)={kidxs, layer, batchseq}" # fmt: skip
            # Store
            vars_explained.append(explained_variances)
            mses.append(sq_errs / d_model)
            error_norms.append(sq_errs.sqrt())
        # Done
        return vars_explained, mses, error_norms # fmt: skip

    def calculate_pca_data(
            self,
            datasets: List[torch.Tensor, "layer batchseq d_model"] | List[torch.Tensor, "ks layer batchseq d_model"]
        ) -> Tuple[
        List[Float[torch.Tensor, "layer batchseq d_model"] | Float[torch.Tensor, "ks layer batchseq d_model"]],
        List[Float[torch.Tensor, "layer 1 d_model"] | Float[torch.Tensor, "ks layer 1 d_model"]],
        List[Float[torch.Tensor, "layer d_model d_model"] | Float[torch.Tensor, "ks layer d_model d_model"]],
        List[Float[torch.Tensor, "layer d_model"] | Float[torch.Tensor, "ks layer d_model"]]
    ]:
        """
        Get the data we will need for our analysis of the PCs. Basically,
        ```
        sae_ins_pca_eigenvectors: Float[torch.Tensor, "layer d_model d_model"]
        sae_ins_pca_eigenvalues: Float[torch.Tensor, "layer d_model"]
        ...
        ```
        """
        means: List[Float[torch.Tensor, "layer 1 d_model"] | Float[torch.Tensor, "ks layer 1 d_model"]] = [] # fmt: skip
        centered_datasets: List[Float[torch.Tensor, "layer batchseq d_model"] | Float[torch.Tensor, "ks layer batchseq d_model"]] = [] # fmt: skip
        eigenvectors: List[Float[torch.Tensor, "layer d_model d_model"] | Float[torch.Tensor, "ks layer d_model d_model"]] = [] # fmt: skip
        eigenvalues: List[Float[torch.Tensor, "layer d_model"] | Float[torch.Tensor, "ks layer d_model"]] = [] # fmt: skip
        for dataset in datasets:
            per_k: bool = dataset.ndim == 4
            assert per_k or dataset.ndim == 3, f"dataset.ndim={dataset.ndim} must be 3 or 4"
            if not per_k:
                dataset = dataset.unsqueeze(0)
            assert dataset.ndim == 4, f"dataset.ndim={dataset.ndim} must be 4, shape={dataset.shape}"
            # Get the means
            batchseq_dim = 2 # NOTE: we unsqueezed the dataset above if necessary
            mean = dataset.mean(dim=batchseq_dim, keepdim=True)
            means.append(mean if per_k else mean.squeeze(0))

            # Get the cov
            centered = dataset - mean
            centered_datasets.append(centered if per_k else centered.squeeze(0))
            batchseq = centered.shape[batchseq_dim]
            assert batchseq > 1, f"shape={centered.shape} must have batchseq > 1"
            reduction_pattern = "ks layer batchseq dim1, ks layer batchseq dim2 -> ks layer dim1 dim2"
            cov = einops.einsum(centered, centered, reduction_pattern).cpu()
            cov = cov / (batchseq - 1)
            # Get the eigendecomposition to get the principal components
            eigendecomposition = torch.linalg.eigh(cov)
            evcs = eigendecomposition.eigenvectors
            evals = eigendecomposition.eigenvalues
            eigenvectors.append(evcs if per_k else evcs.squeeze(0))
            eigenvalues.append(evals if per_k else evals.squeeze(0))
        assert len(centered_datasets) == len(means) == len(eigenvectors) == len(eigenvalues) == len(datasets), f"len(centered_datasets)={len(centered_datasets)} must be len(means)={len(means)} == len(eigenvectors)={len(eigenvectors)} == len(eigenvalues)={len(eigenvalues)} == len(datasets)={len(datasets)}" # fmt: skip
        return (
            centered_datasets,
            means,
            eigenvectors,
            eigenvalues
        )

    def flatten(self) -> FlattenedExtractedActivations:
        # 1. Reshape to ignore batch dimension
        flatten_pattern = "layer batch seq d_model -> layer (batch seq) d_model"
        flatten_pattern_per_k = "ks layer batch seq d_model -> ks layer (batch seq) d_model"
        flatten_func = lambda x: einops.rearrange(x, flatten_pattern)
        flatten_func_per_k = lambda x: einops.rearrange(x, flatten_pattern_per_k)
        datasets = [
            flatten_func(self.sae_ins),
            flatten_func_per_k(self.sae_outs_per_k),
            flatten_func(self.ln2s),
            flatten_func_per_k(self.ln2s_saed_per_k),
        ]
        assert len(datasets) == 4
        
        # 2. Calculate PCA data
        # NOTE: SOME of these are per-k and SOME are NOT per-k
        (
            centered_datasets,
            means,
            eigenvectors,
            eigenvalues
        ) = self.calculate_pca_data(datasets)
        assert len(centered_datasets) == len(means) == len(eigenvectors) == len(eigenvalues) == len(datasets) == 4, f"len(centered_datasets)={len(centered_datasets)} must be len(means)={len(means)} == len(eigenvectors)={len(eigenvectors)} == len(eigenvalues)={len(eigenvalues)} == len(datasets)={len(datasets)}" # fmt: skip

        # 3. Calculate error data
        # NOTE these are ALL per-k
        (
            vars_explained,
            mses,
            error_norms
        ) = self.calculate_error_data(datasets)
        assert len(vars_explained) == len(mses) == len(error_norms) == 2, f"len(vars_explained)={len(vars_explained)} must be len(mses)={len(mses)} == len(error_norms)={len(error_norms)} == 2" # fmt: skip


        return FlattenedExtractedActivations(
            # Data
            sae_ins=datasets[0],
            sae_outs_per_k=datasets[1],
            ln2s=datasets[2],
            ln2s_saed_per_k=datasets[3],

            # Centered data
            sae_ins_centered=centered_datasets[0],
            sae_outs_per_k_centered=centered_datasets[1],
            ln2s_centered=centered_datasets[2],
            ln2s_saed_per_k_centered=centered_datasets[3],

            # Means
            sae_ins_means=means[0],
            sae_outs_per_k_means=means[1],
            ln2s_means=means[2],
            ln2s_saed_per_k_means=means[3],

            # Errors etc...
            #
            # When applied on residual stream
            res_sae_var_explained=vars_explained[0],
            res_sae_mse=mses[0],
            res_sae_error_norms=error_norms[0],

            # Same but for the ln2
            ln2_sae_var_explained=vars_explained[1],
            ln2_sae_mse=mses[1],
            ln2_sae_error_norms=error_norms[1],

            # PCA components, etc...
            #
            # Res with nothing applied
            sae_ins_pca_eigenvectors=eigenvectors[0],
            sae_ins_pca_eigenvalues=eigenvalues[0],
            # Res with sae applied
            sae_outs_per_k_pca_eigenvectors=eigenvectors[1],
            sae_outs_per_k_pca_eigenvalues=eigenvalues[1],
            # ln2 with nothing applied
            ln2s_pca_eigenvectors=eigenvectors[2],
            ln2s_pca_eigenvalues=eigenvalues[2],
            # ln2 with sae applied
            ln2s_saed_per_k_pca_eigenvectors=eigenvectors[3],
            ln2s_saed_per_k_pca_eigenvalues=eigenvalues[3],
        )