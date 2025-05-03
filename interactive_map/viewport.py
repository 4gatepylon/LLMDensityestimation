from __future__ import annotations

import pydantic
import torch
from torch import Tensor
from jaxtyping import Float, Int
from typing import List, Optional, Literal


class Viewport2D(pydantic.BaseModel):
    """
    A viewport is a 2D projection matrix and mean shift.
    You can think of it like a "flat camera". The way things
    get projected is the following:
    1. You define a grid of pixels in the projective plane.
    2. You take in your activations/locations in higher dimensional space
        and then you mean-shift them if the mean is applicable and then
        project them onto the viewport, binning them into the pixels.
    3. In this bins, you write some optional values (these could be errors
        or something else).
    
    The output here is a list of x, y coordinates and values to put in those
    bins.
    """
    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    # A projection matrix and mean shift that you can use to project activations
    # Using orthogonal projection.
    projection_matrix: Float[Tensor, "n_components n_features"]
    mean_shift: Optional[Float[Tensor, "n_features"]] = None
    using_mean_shift: bool = True

    # NOTE: viewports have to be 2D for now, but this may change
    # We define limits; but they can be None if you want to specify "no limit"
    # (though in that case the behavior of plotting functionality will be DEFINED
    # by whomever the user/client is)
    x_left: Optional[float] = None
    x_right: Optional[float] = None
    y_bottom: Optional[float] = None
    y_top: Optional[float] = None

    n_bins_x: Optional[int] = None
    n_bins_y: Optional[int] = None

    def project_onto_viewport(
            self,
            locations: Float[Tensor, "n_samples n_features"],
            values: Optional[Float[Tensor, "n_samples"]] = None,
            component_idxs: List[int] = [0, 1],
            use_cuda: bool = False,
            return_raw: bool = False,
            clamp: bool = True,
        ) -> Float[Tensor, "n_samples 3"]:
        """
        Return the values of projection onto a viewport. This is defined in the
        docstring for this class. If you don't provide values, values are
        assumed to always be 1.0.
        
        If you pass `return_raw=True`, then the raw x/y location coordinates
        in that basis are returned.

        If you pass `clamp=True`, then the values are clamped to the viewport
        and otherwise they are REMOVED.
        """
        if locations.ndim != 2:
            raise ValueError(f"Locations must be a 2D tensor, got shape {locations.shape}")
        if len(component_idxs) != 2:
            raise ValueError("You must provide exactly 2 component idxs")
        if not all(0 <= i < locations.shape[1] for i in component_idxs):
            raise ValueError("All component idxs must be less than the number of features")
        assert values.ndim == 1
        assert values.shape[0] == locations.shape[0]

        if use_cuda:
            locations = locations.cuda()
            projection_matrix = self.projection_matrix.cuda()
            mean_shift = self.mean_shift.cuda()
        
        # Now deal with the the fact that we might have no values
        if values is None:
            values = torch.ones_like(locations[:, 0])
        assert values.ndim == 1
        assert values.shape[0] == locations.shape[0]

        # Center the data
        if self.using_mean_shift:
            locations = locations - mean_shift
        projected = torch.matmul(locations, projection_matrix[:, component_idxs])
        # So now we've taken the dot product and it's time to normalize
        # project V onto direction W means take (V dot W) / |W||V| * |V| * W / |W|
        # but we want the magnitude which is (V dot W) / |W| => just divide by the norm of W
        # (which SHOULD be 1 here but just in case...)
        Ws = projection_matrix[:, component_idxs].norm(dim=0)
        # activations = activations.to(activations_device) # I don't think this really matters?
        # pca_mean = pca_mean.to(pca_mean_device)
        # pca_components = pca_components.to(pca_components_device)
        xy_coordinates = projected / Ws
        assert xy_coordinates.ndim == 2
        assert xy_coordinates.shape[0] == locations.shape[0]
        assert xy_coordinates.shape[1] == 2
        if return_raw:
            return xy_coordinates
        else:
            if self.n_bins_x is None or self.n_bins_y is None:
                raise ValueError("You must specify n_bins_x and n_bins_y")
            if self.x_left is None or self.x_right is None or self.y_bottom is None or self.y_top is None:
                raise ValueError("You must specify x_left, x_right, y_bottom, y_top")
            
            # 1. Deal with clamping issues
            x_too_large = xy_coordinates[:, 0] > self.x_right
            x_too_small = xy_coordinates[:, 0] < self.x_left
            y_too_large = xy_coordinates[:, 1] > self.y_top
            y_too_small = xy_coordinates[:, 1] < self.y_bottom
            if not clamp:
                exclude_points = x_too_large | x_too_small | y_too_large | y_too_small
                if exclude_points.sum() >= xy_coordinates.shape[0]:
                    raise ValueError("No points left after clamping")
                xy_coordinates = xy_coordinates[~exclude_points, :]
                values = values[~exclude_points]
            else:
                # Clamp to the max, min, etc...
                xy_coordinates[x_too_large, 0] = self.x_right
                xy_coordinates[x_too_small, 0] = self.x_left
                xy_coordinates[y_too_large, 1] = self.y_top
                xy_coordinates[y_too_small, 1] = self.y_bottom

            # 2. Digitize/Bin the data into pixels (mainly by Claude here)
            # Bin the data into a 2D histogram
            x_bins = torch.linspace(self.x_left, self.x_right, self.n_bins_x + 1, device=xy_coordinates.device)
            y_bins = torch.linspace(self.y_bottom, self.y_top, self.n_bins_y + 1, device=xy_coordinates.device)
            
            # Find which bin each point belongs to
            x_indices = torch.bucketize(xy_coordinates[:, 0], x_bins) - 1
            y_indices = torch.bucketize(xy_coordinates[:, 1], y_bins) - 1
            
            # we should already be clamped, but just in case
            assert (x_indices >= 0).all()
            assert (y_indices >= 0).all()
            assert (x_indices < self.n_bins_x).all()
            assert (y_indices < self.n_bins_y).all()
            
            # Don't create the bin yet, we are just going to return the indices
            xy_indices = torch.cat([x_indices[:, None], y_indices[:, None]], dim=1)
            assert xy_indices.ndim == 2
            assert xy_indices.shape[0] == locations.shape[0]
            assert xy_indices.shape[1] == 2
            return xy_indices, values
    
    def get_grid(
            self,
            indices: Int[Tensor, "n_samples 2"],
            values: Float[Tensor, "n_samples"],
            reduction: Literal["sum"] = "sum",
        ) -> Float[Tensor, "n_bins_y n_bins_x"]:
        """
        Generate a grid of the viewport. You must provide some kind of
        reduction method. Only sum is supported for now.
        """
        values = values.to(indices.device)
        grid = torch.zeros((self.n_bins_y, self.n_bins_x), device=indices.device)
        grid[indices[:, 1], indices[:, 0]] = values
        if reduction == "sum":
            return grid
        else:
            raise NotImplementedError(f"Unsupported reduction method: {reduction}")


class PCAViewport(Viewport2D):
    """
    This is the same as a regular Viewport2D but it also stores its eigenvalues.
    It is guaranteed that the the eigenvalues and eigenvectors will be sorted
    by descending eigenvalue.
    """
    eigenvalues: Float[Tensor, "n_components"]

    @staticmethod
    def calculate_pca(
        activations: Float[Tensor, "n_samples n_features"],
        use_cuda: bool = False,
    ) -> PCAViewport:
        """Return and useable viewport for the PCA components."""
        # Center the data
        device = activations.device
        if use_cuda:
            activations = activations.cuda()
        activations_mean = activations.mean(dim=0, keepdim=True)
        activations = activations - activations_mean

        # Calculate covariance matrix
        n_samples = activations.shape[0]
        cov_matrix = torch.matmul(activations.T, activations) / (n_samples - 1)
        # Get eigenvalues and eigenvectors
        eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrix)
        # Sort eigenvectors by eigenvalues in descending order
        idx = torch.argsort(eigenvalues, descending=True)
        # NOTE from `eigenvectors will have the same dtype as A and will contain the eigenvectors as its columns.`
        # that we have: https://pytorch.org/docs/stable/generated/torch.linalg.eigh.html
        eigenvectors = eigenvectors[:, idx]
        eigenvalues = eigenvalues[idx]
        # Set the devices
        eigenvectors = eigenvectors.to(device)
        eigenvalues = eigenvalues.to(device)
        activations_mean = activations_mean.flatten().to(device)
        # Return the object
        return PCAViewport(
            # Put the eigenvectors as the ROWS so that when we left multiply
            # (your locations come FROM the right) we get the correct projection.
            projection_matrix=eigenvectors.T,
            eigenvalues=eigenvalues,
            mean_shift=activations_mean,

            # Unbounded for now
            x_left=None,
            x_right=None,
            y_bottom=None,
            y_top=None,
        )