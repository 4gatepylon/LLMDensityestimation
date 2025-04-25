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

    def calculate_error_data(self, datasets: List[torch.Tensor, "layer batchseq d_model"]) -> Tuple[
        List[Float[torch.Tensor, "layer batchseq"]],
        List[Float[torch.Tensor, "layer batchseq"]],
        List[Float[torch.Tensor, "layer batchseq"]],
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
        raise NotImplementedError("Not implemented need to support multiple ks")
        assert len(datasets) == 4, f"len(datasets)={len(datasets)} must be 4"
        data_pairs = [
            # saes_in, saes_out
            (datasets[0], datasets[1]),
            # ln2s, ln2s_saed
            (datasets[2], datasets[3]),
        ]
        layer, batchseq, d_model = datasets[0].shape
        vars_explained = []
        mses = []
        error_norms = []
        for sae_in, sae_out in data_pairs:
            # Calculate
            sq_errs = (sae_in - sae_out).pow(2).sum(dim=-1) # norms of errors
            assert sq_errs.shape == (layer, batchseq), f"sq_errs.shape={sq_errs.shape} must be (layer, batchseq)={layer, batchseq}" # fmt: skip
            var = (sae_in - sae_in.mean(dim=0)).pow(2).sum(dim=-1) # variance of 1st dim then sum those
            assert var.shape == (layer, batchseq), f"var.shape={var.shape} must be (layer, batchseq)={layer, batchseq}" # fmt: skip
            explained_variances = 1 - sq_errs / var
            assert explained_variances.shape == (layer, batchseq), f"explained_variances.shape={explained_variances.shape} must be (layer, batchseq)={layer, batchseq}" # fmt: skip
            # Store
            vars_explained.append(explained_variances)
            mses.append(sq_errs / d_model)
            error_norms.append(sq_errs.sqrt())
        # Done
        return vars_explained, mses, error_norms # fmt: skip

    def calculate_pca_data(self, datasets: List[torch.Tensor, "layer batchseq d_model"]) -> Tuple[
        List[Float[torch.Tensor, "layer batchseq d_model"]],
        List[Float[torch.Tensor, "layer 1 d_model"]],
        List[Float[torch.Tensor, "layer d_model d_model"]],
        List[Float[torch.Tensor, "layer d_model"]]
    ]:
        """
        Get the data we will need for our analysis of the PCs. Basically,
        ```
        sae_ins_pca_eigenvectors: Float[torch.Tensor, "layer d_model d_model"]
        sae_ins_pca_eigenvalues: Float[torch.Tensor, "layer d_model"]
        ...
        ```
        """
        raise NotImplementedError("Not implemented need to support multiple ks")
        means: List[Float[torch.Tensor, "layer 1 d_model"]] = []
        centered_datasets: List[Float[torch.Tensor, "layer batchseq d_model"]] = []
        eigenvectors: List[Float[torch.Tensor, "layer d_model d_model"]] = []
        eigenvalues: List[Float[torch.Tensor, "layer d_model"]] = []
        for dataset in datasets:
            # Get the means
            mean = dataset.mean(dim=1, keepdim=True)
            means.append(mean)

            # Get the cov
            centered = dataset - mean
            centered_datasets.append(centered)
            batchseq = centered.shape[1]
            assert batchseq > 1, f"shape={centered.shape} must have batchseq > 1"
            reduction_pattern = "layer batchseq dim1, layer batchseq dim2 -> layer dim1 dim2"
            cov = einops.einsum(centered, centered, reduction_pattern).cpu()
            cov = cov / (batchseq - 1)
            # Get the eigendecomposition to get the principal components
            eigendecomposition = torch.linalg.eigh(cov)
            
            eigenvectors.append(eigendecomposition.eigenvectors)
            eigenvalues.append(eigendecomposition.eigenvalues)
        assert len(centered_datasets) == len(means) == len(eigenvectors) == len(eigenvalues) == len(datasets), f"len(centered_datasets)={len(centered_datasets)} must be len(means)={len(means)} == len(eigenvectors)={len(eigenvectors)} == len(eigenvalues)={len(eigenvalues)} == len(datasets)={len(datasets)}" # fmt: skip
        return (
            centered_datasets,
            means,
            eigenvectors,
            eigenvalues
        )

    def flatten(self) -> FlattenedExtractedActivations:
        # 1. Reshape to ignore batch dimension
        raise NotImplementedError("Not implemented need to support multiple ks")
        flatten_pattern = "layer batch seq d_model -> layer (batch seq) d_model"
        flatten_func = lambda x: einops.rearrange(x, flatten_pattern)
        datasets = []
        for unflat in [self.sae_ins, self.sae_outs, self.ln2s, self.ln2s_saed]:
            datasets.append(flatten_func(unflat))
        assert len(datasets) == 4
        
        # 2. Calculate PCA data
        (
            centered_datasets,
            means,
            eigenvectors,
            eigenvalues
        ) = self.calculate_pca_data(datasets)
        assert len(centered_datasets) == len(means) == len(eigenvectors) == len(eigenvalues) == len(datasets) == 4, f"len(centered_datasets)={len(centered_datasets)} must be len(means)={len(means)} == len(eigenvectors)={len(eigenvectors)} == len(eigenvalues)={len(eigenvalues)} == len(datasets)={len(datasets)}" # fmt: skip

        # 3. Calculate error data
        (
            vars_explained,
            mses,
            error_norms
        ) = self.calculate_error_data(datasets)
        assert len(vars_explained) == len(mses) == len(error_norms) == 2, f"len(vars_explained)={len(vars_explained)} must be len(mses)={len(mses)} == len(error_norms)={len(error_norms)} == 2" # fmt: skip


        return FlattenedExtractedActivations(
            # Data
            sae_ins=datasets[0],
            sae_outs=datasets[1],
            ln2s=datasets[2],
            ln2s_saed=datasets[3],

            # Centered data
            sae_ins_centered=centered_datasets[0],
            sae_outs_centered=centered_datasets[1],
            ln2s_centered=centered_datasets[2],
            ln2s_saed_centered=centered_datasets[3],

            # Means
            sae_ins_means=means[0],
            sae_outs_means=means[1],
            ln2s_means=means[2],
            ln2s_saed_means=means[3],

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
            sae_outs_pca_eigenvectors=eigenvectors[1],
            sae_outs_pca_eigenvalues=eigenvalues[1],
            # ln2 with nothing applied
            ln2s_pca_eigenvectors=eigenvectors[2],
            ln2s_pca_eigenvalues=eigenvalues[2],
            # ln2 with sae applied
            ln2s_saed_pca_eigenvectors=eigenvectors[3],
            ln2s_saed_pca_eigenvalues=eigenvalues[3],
        )