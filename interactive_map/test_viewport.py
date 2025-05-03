from __future__ import annotations
from pathlib import Path
import sys
import pytest
import torch
import numpy as np

try:
    import viewport
except ImportError:
    sys.path.append(Path(__file__).parent.as_posix())
    print("sys path is now 2:", sys.path) # Printing fixed this?
    try:
        import viewport
    except ImportError:
        try:
            import interactive_map.viewport as viewport
        except ImportError:
            sys.path.append(Path(__file__).parent.parent.as_posix())
            print("sys path is now 3:", sys.path) # Printing fixed this?
            import interactive_map.viewport as viewport

Viewport2D, PCAViewport = viewport.Viewport2D, viewport.PCAViewport


@pytest.fixture
def identity_viewport():
    """Create a simple identity projection viewport for testing"""
    identity_projection = torch.eye(2)
    return Viewport2D(
        projection_matrix=identity_projection,
        mean_shift=torch.zeros(2),
        using_mean_shift=True,
        x_left=-5.0,
        x_right=5.0,
        y_bottom=-5.0,
        y_top=5.0,
        n_bins_x=10,
        n_bins_y=10
    )


# Set random seed for all tests
@pytest.fixture(autouse=True)
def set_random_seed():
    torch.manual_seed(42)
    np.random.seed(42)


class TestViewport2D:
    """
    By Claude.
    """
    
    def test_simple_projection(self, identity_viewport):
        """Test simple projection with identity matrix"""
        # Create 4 points at the corners of a square
        points = torch.tensor([
            [2.0, 2.0],   # top-right
            [-2.0, 2.0],  # top-left
            [-2.0, -2.0], # bottom-left
            [2.0, -2.0],  # bottom-right
        ])
        values = torch.ones(4)
        
        # Project points
        indices, projected_values = identity_viewport.project_onto_viewport(
            points, values, component_idxs=[0, 1]
        )
        
        # Expected bin indices (convert coordinates to bin indices)
        # For x: [-5, 5] -> [0, 9] bins
        # For y: [-5, 5] -> [0, 9] bins
        # 2.0 should map to bin 7, -2.0 should map to bin 3
        expected_indices = torch.tensor([
            [7, 7],  # top-right
            [3, 7],  # top-left
            [3, 3],  # bottom-left
            [7, 3],  # bottom-right
        ])
        
        assert torch.all(indices == expected_indices)
        assert torch.all(projected_values == values)
        
        # Test the resulting grid
        grid = identity_viewport.get_grid(indices, projected_values)
        
        # Expected grid: zeros except at the four corners
        expected_grid = torch.zeros((10, 10))
        expected_grid[7, 7] = 1.0  # top-right (y, x)
        expected_grid[7, 3] = 1.0  # top-left (y, x)
        expected_grid[3, 3] = 1.0  # bottom-left (y, x)
        expected_grid[3, 7] = 1.0  # bottom-right (y, x)
        
        assert torch.allclose(grid, expected_grid)
    
    def test_clamping(self, identity_viewport):
        """Test that points outside the viewport are properly clamped"""
        # Create points outside the viewport bounds
        points = torch.tensor([
            [10.0, 3.0],   # right edge
            [-10.0, 3.0],  # left edge
            [3.0, 10.0],   # top edge
            [3.0, -10.0],  # bottom edge
        ])
        values = torch.ones(4)
        
        # Project with clamping
        indices, projected_values = identity_viewport.project_onto_viewport(
            points, values, component_idxs=[0, 1], clamp=True
        )
        
        # Expected indices after clamping
        expected_indices = torch.tensor([
            [9, 6],  # right edge (clamped to x_right)
            [0, 6],  # left edge (clamped to x_left)
            [6, 9],  # top edge (clamped to y_top)
            [6, 0],  # bottom edge (clamped to y_bottom)
        ])
        
        assert torch.all(indices == expected_indices)
    
    def test_center_loading(self, identity_viewport):
        """Test that a Gaussian distribution loads more density in the center"""
        # Generate random Gaussian data
        n_samples = 1000
        points = torch.randn(n_samples, 2)
        values = torch.ones(n_samples)
        
        # Project points
        indices, projected_values = identity_viewport.project_onto_viewport(
            points, values, component_idxs=[0, 1]
        )
        
        # Create grid
        grid = identity_viewport.get_grid(indices, projected_values)
        
        # The center bins should have higher counts than edge bins
        center_bins = grid[4:6, 4:6]
        edge_bins = torch.cat([
            grid[0:2, :].flatten(),
            grid[-2:, :].flatten(),
            grid[:, 0:2].flatten(),
            grid[:, -2:].flatten()
        ])
        
        assert center_bins.sum() > edge_bins.mean() * 4
    
    def test_custom_projection(self):
        """Test a non-identity projection matrix"""
        # Create a projection that swaps x and y and scales
        projection = torch.tensor([
            [0.0, 2.0],  # y-axis becomes x-axis, doubled
            [0.5, 0.0]   # x-axis becomes y-axis, halved
        ])
        
        custom_viewport = Viewport2D(
            projection_matrix=projection,
            mean_shift=torch.zeros(2),
            using_mean_shift=True,
            x_left=-5.0,
            x_right=5.0,
            y_bottom=-5.0,
            y_top=5.0,
            n_bins_x=10,
            n_bins_y=10
        )
        
        # Point at (2, 4) should project to (8, 1)
        points = torch.tensor([[2.0, 4.0]])
        values = torch.ones(1)
        
        # Get raw projection
        projected = custom_viewport.project_onto_viewport(
            points, values, component_idxs=[0, 1], return_raw=True
        )
        
        # Expected: y becomes x and x becomes y
        # [2, 4] -> [4*2, 2*0.5] -> [8, 1]
        expected = torch.tensor([[8.0, 1.0]])
        assert torch.allclose(projected, expected)


class TestPCAViewport:
    """
    By Claude.
    """
    def test_pca_calculation(self):
        """Test PCA calculation with simple synthetic data"""
        # Create synthetic data with known structure
        # Principal direction along [1, 0.5]
        n_samples = 1000
        noise = 0.1
        
        # Generate points along the line y = 0.5x + noise
        x = torch.linspace(-5, 5, n_samples)
        y = 0.5 * x + noise * torch.randn(n_samples)
        data = torch.stack([x, y], dim=1)
        
        # Calculate PCA
        pca_viewport = PCAViewport.calculate_pca(data)
        
        # The first principal component should be close to [1, 0.5] normalized
        pc1 = pca_viewport.projection_matrix[0]
        norm_factor = torch.norm(torch.tensor([1.0, 0.5]))
        expected_pc1 = torch.tensor([1.0, 0.5]) / norm_factor
        
        # Allow for sign flip
        is_close = torch.allclose(pc1, expected_pc1, atol=0.2) or \
                  torch.allclose(pc1, -expected_pc1, atol=0.2)
        
        assert is_close, f"PC1 {pc1} is not close to expected {expected_pc1}"
        
        # First eigenvalue should be larger than second
        assert pca_viewport.eigenvalues[0] > pca_viewport.eigenvalues[1]
    
    def test_pca_projection(self):
        """Test projection with PCA viewport"""
        # Create 2D data with primary variance along y = x
        n_samples = 1000
        x = torch.linspace(-3, 3, n_samples)
        y = x + 0.1 * torch.randn(n_samples)
        data = torch.stack([x, y], dim=1)
        
        # Calculate PCA
        pca_viewport = PCAViewport.calculate_pca(data)
        
        # Configure viewport bounds
        pca_viewport.x_left = -5
        pca_viewport.x_right = 5
        pca_viewport.y_bottom = -5
        pca_viewport.y_top = 5
        pca_viewport.n_bins_x = 10
        pca_viewport.n_bins_y = 10
        
        # Project the data
        indices, values = pca_viewport.project_onto_viewport(
            data, torch.ones(n_samples)
        )
        
        grid = pca_viewport.get_grid(indices, values)
        
        # Since the data is mostly along the first principal component,
        # we expect most points to be distributed along a diagonal line
        # Check that the grid has a non-zero diagonal pattern
        diag_sum = sum(grid[i, i] for i in range(min(10, 10)))
        total_sum = grid.sum()
        
        # At least 30% of points should be along the diagonal
        assert diag_sum / total_sum > 0.3