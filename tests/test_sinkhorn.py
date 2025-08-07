#!/usr/bin/env python3
"""
Test cases for the sinkhorn_normalize function.

This module tests the Sinkhorn-Knopp normalization algorithm that makes a matrix
symmetric with row/col sums matching target degrees.
"""

import numpy as np
import pytest
import sys
import os

# Add the code directory to the path more reliably
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'code'))

from methods import sinkhorn_normalize


class TestSinkhornNormalize:
    """Test cases for sinkhorn_normalize function."""
    
    def test_basic_symmetric_matrix(self):
        """Test with a simple symmetric matrix."""
        # Create a simple 3x3 symmetric matrix
        W = np.array([
            [0, 1, 2],
            [1, 0, 3],
            [2, 3, 0]
        ], dtype=float)
        
        target_degrees = np.array([3, 4, 5])
        
        W_norm = sinkhorn_normalize(W, target_degrees, verbose=False)
        
        # Check basic properties
        assert W_norm.shape == (3, 3)
        assert np.allclose(W_norm, W_norm.T, atol=1e-8), "Matrix should be symmetric"
        assert np.allclose(np.diag(W_norm), 0, atol=1e-8), "Diagonal should be zero"
        
        # Check that row/col sums are close to target degrees
        row_sums = W_norm.sum(axis=1)
        col_sums = W_norm.sum(axis=0)
        
        assert np.allclose(row_sums, target_degrees, atol=1e-3), f"Row sums {row_sums} should match target {target_degrees}"
        assert np.allclose(col_sums, target_degrees, atol=1e-3), f"Col sums {col_sums} should match target {target_degrees}"
    
    def test_uniform_target_degrees(self):
        """Test with uniform target degrees."""
        n = 5
        W = np.random.rand(n, n)
        W = (W + W.T) / 2  # Make symmetric
        np.fill_diagonal(W, 0)  # Zero diagonal
        
        target_degrees = np.ones(n) * 2.0  # All nodes should have degree 2
        
        W_norm = sinkhorn_normalize(W, target_degrees, verbose=False)
        
        # Check properties
        assert W_norm.shape == (n, n)
        assert np.allclose(W_norm, W_norm.T, atol=1e-8)
        assert np.allclose(np.diag(W_norm), 0, atol=1e-8)
        
        # Check that all row/col sums are close to 2.0
        row_sums = W_norm.sum(axis=1)
        assert np.allclose(row_sums, 2.0, atol=1e-3), f"All row sums should be 2.0, got {row_sums}"
    
    def test_heterogeneous_target_degrees(self):
        """Test with different target degrees for each node."""
        n = 4
        W = np.random.rand(n, n)
        W = (W + W.T) / 2
        np.fill_diagonal(W, 0)
        
        target_degrees = np.array([1.0, 2.0, 3.0, 4.0])
        
        W_norm = sinkhorn_normalize(W, target_degrees, verbose=False)
        
        # Check properties
        assert W_norm.shape == (n, n)
        assert np.allclose(W_norm, W_norm.T, atol=1e-8)
        assert np.allclose(np.diag(W_norm), 0, atol=1e-8)
        
        # Check that row/col sums match target degrees
        row_sums = W_norm.sum(axis=1)
        assert np.allclose(row_sums, target_degrees, atol=1e-3), f"Row sums {row_sums} should match target {target_degrees}"
    
    def test_sparse_matrix(self):
        """Test with a sparse matrix (many zeros)."""
        n = 6
        W = np.zeros((n, n))
        # Add some random edges
        edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (0, 5)]
        for i, j in edges:
            W[i, j] = W[j, i] = np.random.rand()
        
        target_degrees = np.array([2.0, 2.0, 2.0, 2.0, 2.0, 2.0])
        
        W_norm = sinkhorn_normalize(W, target_degrees, verbose=False)
        
        # Check properties
        assert W_norm.shape == (n, n)
        assert np.allclose(W_norm, W_norm.T, atol=1e-8)
        assert np.allclose(np.diag(W_norm), 0, atol=1e-8)
        
        # Check that row/col sums match target degrees
        row_sums = W_norm.sum(axis=1)
        assert np.allclose(row_sums, target_degrees, atol=1e-3), f"Row sums {row_sums} should match target {target_degrees}"
    
    def test_convergence_with_different_tolerances(self):
        """Test convergence with different tolerance values."""
        n = 4
        W = np.random.rand(n, n)
        W = (W + W.T) / 2
        np.fill_diagonal(W, 0)
        
        target_degrees = np.array([1.5, 2.5, 3.5, 4.5])
        
        # Test with different tolerances
        for tol in [1e-3, 1e-4, 1e-5]:
            W_norm = sinkhorn_normalize(W, target_degrees, tol=tol, verbose=False)
            
            # Check that the error is within the tolerance
            row_sums = W_norm.sum(axis=1)
            max_error = np.max(np.abs(row_sums - target_degrees))
            assert max_error <= tol, f"Error {max_error} should be <= tolerance {tol}"
    
    def test_max_iterations_limit(self):
        """Test behavior when convergence is not reached within max_iterations."""
        n = 3
        # Create a problematic matrix that might not converge easily
        W = np.array([
            [0, 1e-10, 1e-10],
            [1e-10, 0, 1e-10],
            [1e-10, 1e-10, 0]
        ])
        
        target_degrees = np.array([1.0, 1.0, 1.0])
        
        # Test with very low max_iterations
        W_norm = sinkhorn_normalize(W, target_degrees, max_iter=5, verbose=False)
        
        # Should still return a valid matrix
        assert W_norm.shape == (n, n)
        assert np.allclose(W_norm, W_norm.T, atol=1e-8)
        assert np.allclose(np.diag(W_norm), 0, atol=1e-8)
    
    def test_edge_cases(self):
        """Test edge cases and potential issues."""
        # Test with very small matrix
        W = np.array([[0, 0.5], [0.5, 0]])
        target_degrees = np.array([1.0, 1.0])
        
        W_norm = sinkhorn_normalize(W, target_degrees, verbose=False)
        assert W_norm.shape == (2, 2)
        assert np.allclose(W_norm.sum(axis=1), target_degrees, atol=1e-3)
        
        # Test with all-zero matrix
        W = np.zeros((3, 3))
        target_degrees = np.array([1.0, 1.0, 1.0])
        
        W_norm = sinkhorn_normalize(W, target_degrees, verbose=False)
        assert W_norm.shape == (3, 3)
        # Should still try to match target degrees even with zero input
    
    def test_networkx_compatibility(self):
        """Test that the normalized matrix can be used with NetworkX."""
        try:
            import networkx as nx
            
            n = 5
            W = np.random.rand(n, n)
            W = (W + W.T) / 2
            np.fill_diagonal(W, 0)
            
            target_degrees = np.array([2.0, 2.0, 2.0, 2.0, 2.0])
            
            W_norm = sinkhorn_normalize(W, target_degrees, verbose=False)
            
            # Convert to NetworkX graph
            G = nx.from_numpy_array(W_norm)
            
            # Check that the graph has the expected properties
            assert G.number_of_nodes() == n
            
            # Check for self-loops (should be none since diagonal is zero)
            self_loops = list(nx.selfloop_edges(G))
            assert len(self_loops) == 0, "No self-loops should exist"
            
            # Check weighted degrees (sum of edge weights)
            weighted_degrees = [G.degree(i, weight='weight') for i in range(n)]
            assert np.allclose(weighted_degrees, target_degrees, atol=1e-3), f"Weighted degrees {weighted_degrees} should match target {target_degrees}"
            
            # Also test with binary conversion
            # Convert soft matrix to binary using threshold
            W_binary = (W_norm > 0.1).astype(float)  # Threshold to get binary edges
            G_binary = nx.from_numpy_array(W_binary)
            
            # Check binary degrees (number of edges)
            binary_degrees = [G_binary.degree(i) for i in range(n)]
            print(f"Binary degrees: {binary_degrees}")
            print(f"Target degrees: {target_degrees}")
            
        except ImportError:
            pytest.skip("NetworkX not available")
    
    def test_stochastic_block_model_compatibility(self):
        """Test with matrices that could come from stochastic block models."""
        # Simulate a 2-block SBM matrix
        n1, n2 = 3, 2
        n = n1 + n2
        
        # Create block structure
        W = np.zeros((n, n))
        
        # Within-block connections (higher probability)
        W[:n1, :n1] = np.random.rand(n1, n1) * 0.8
        W[n1:, n1:] = np.random.rand(n2, n2) * 0.8
        
        # Between-block connections (lower probability)
        W[:n1, n1:] = np.random.rand(n1, n2) * 0.2
        W[n1:, :n1] = W[:n1, n1:].T
        
        # Make symmetric and zero diagonal
        W = (W + W.T) / 2
        np.fill_diagonal(W, 0)
        
        target_degrees = np.array([2.0, 2.0, 2.0, 1.5, 1.5])
        
        W_norm = sinkhorn_normalize(W, target_degrees, verbose=False)
        
        # Check properties
        assert W_norm.shape == (n, n)
        assert np.allclose(W_norm, W_norm.T, atol=1e-8)
        assert np.allclose(np.diag(W_norm), 0, atol=1e-8)
        
        # Check that row/col sums match target degrees
        row_sums = W_norm.sum(axis=1)
        assert np.allclose(row_sums, target_degrees, atol=1e-3), f"Row sums {row_sums} should match target {target_degrees}"
    
    def test_verbose_output(self):
        """Test that verbose output works correctly."""
        n = 4
        W = np.random.rand(n, n)
        W = (W + W.T) / 2
        np.fill_diagonal(W, 0)
        
        target_degrees = np.array([2.0, 2.0, 2.0, 2.0])
        
        # This should not raise any errors
        W_norm = sinkhorn_normalize(W, target_degrees, verbose=True)
        
        assert W_norm.shape == (n, n)
        assert np.allclose(W_norm, W_norm.T, atol=1e-8)
    
    def test_large_matrix(self):
        """Test with a larger matrix to ensure scalability."""
        n = 20
        W = np.random.rand(n, n)
        W = (W + W.T) / 2
        np.fill_diagonal(W, 0)
        
        target_degrees = np.random.uniform(1.0, 5.0, n)
        
        W_norm = sinkhorn_normalize(W, target_degrees, verbose=False)
        
        # Check properties
        assert W_norm.shape == (n, n)
        assert np.allclose(W_norm, W_norm.T, atol=1e-8)
        assert np.allclose(np.diag(W_norm), 0, atol=1e-8)
        
        # Check that row/col sums match target degrees
        row_sums = W_norm.sum(axis=1)
        assert np.allclose(row_sums, target_degrees, atol=1e-3), f"Row sums {row_sums} should match target {target_degrees}"


def test_sinkhorn_properties():
    """Test that sinkhorn_normalize preserves important properties."""
    # Test that the function preserves the relative structure of the input matrix
    n = 4
    W = np.array([
        [0, 1, 0, 2],
        [1, 0, 3, 0],
        [0, 3, 0, 4],
        [2, 0, 4, 0]
    ], dtype=float)
    
    target_degrees = np.array([3.0, 4.0, 7.0, 6.0])
    
    W_norm = sinkhorn_normalize(W, target_degrees, verbose=False)
    
    # Check that zero entries in input remain very small in output (due to 1e-12 addition)
    zero_mask = (W == 0)
    assert np.all(W_norm[zero_mask] <= 1e-10), "Zero entries in input should remain very small in output"
    
    # Check that non-zero entries in input remain non-zero in output
    nonzero_mask = (W != 0)
    assert np.all(W_norm[nonzero_mask] > 0), "Non-zero entries in input should remain non-zero"
    
    # Check that the relative structure is preserved (larger entries stay larger)
    for i in range(n):
        for j in range(n):
            if i != j:  # Skip diagonal
                if W[i, j] > W[j, i]:
                    assert W_norm[i, j] >= W_norm[j, i], "Relative structure should be preserved"
                elif W[i, j] < W[j, i]:
                    assert W_norm[i, j] <= W_norm[j, i], "Relative structure should be preserved"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"]) 