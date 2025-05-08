import pytest
import numpy as np
import networkx as nx
from scipy.sparse import csr_matrix
import sys
import os

# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from code.data_loader import (
    create_core_fringe_graph,
    create_multi_dorm_core_fringe_graph,
    link_logistic_regression_core_only_auc
)

def create_test_clique_and_tree():
    """Creates a test graph with a 4-node clique and 3 fringe nodes.
    Only core-core and core-fringe edges are observed."""
    G = nx.Graph()
    # Create a 4-node clique (core)
    clique_nodes = [0, 1, 2, 3]
    for i in clique_nodes:
        for j in clique_nodes:
            if i < j:
                G.add_edge(i, j)
    
    # Create 3 fringe nodes, each connected to node 0
    fringe_nodes = [4, 5, 6]
    for f in fringe_nodes:
        G.add_edge(0, f)  # all fringe nodes connected to core node 0
    
    # Convert to adjacency matrix with int32 dtype
    adj_matrix = nx.to_scipy_sparse_array(G, format='csr', dtype=np.int32)
    
    # Create metadata with dorm IDs
    metadata = np.zeros((7, 7))  # 7 nodes, 7 metadata columns
    metadata[clique_nodes, 4] = 1  # dorm_id=1 for core nodes
    metadata[fringe_nodes, 4] = 2  # dorm_id=2 for fringe nodes
    
    return adj_matrix, metadata

def create_test_two_cliques():
    """Creates a test graph with two 4-node cliques, perfectly separable by gender."""
    G = nx.Graph()
    # Create two 4-node cliques
    clique1_nodes = [0, 1, 2, 3]  # core
    clique2_nodes = [4, 5, 6, 7]  # fringe
    
    # Add edges within each clique
    for i in clique1_nodes:
        for j in clique1_nodes:
            if i < j:
                G.add_edge(i, j)
    for i in clique2_nodes:
        for j in clique2_nodes:
            if i < j:
                G.add_edge(i, j)
    
    # Add core-fringe edges
    G.add_edge(0, 4)
    G.add_edge(1, 5)
    G.add_edge(2, 6)
    G.add_edge(3, 7)
    
    # Convert to adjacency matrix with int32 dtype and indices
    adj_matrix = nx.to_scipy_sparse_array(G, format='csr', dtype=np.int32)
    # Ensure indices are int32
    adj_matrix.indices = adj_matrix.indices.astype(np.int32)
    adj_matrix.indptr = adj_matrix.indptr.astype(np.int32)
    
    # Create metadata with dorm IDs and gender
    metadata = np.zeros((8, 7))  # 8 nodes, 7 metadata columns
    metadata[clique1_nodes, 4] = 1  # dorm_id=1 for core
    metadata[clique2_nodes, 4] = 2  # dorm_id=2 for fringe
    
    # Set gender labels to ensure two classes in core
    metadata[[0, 1], 1] = 1  # first two core nodes gender=1
    metadata[[2, 3], 1] = 2  # last two core nodes gender=2
    metadata[clique2_nodes, 1] = 2  # all fringe nodes gender=2
    
    return adj_matrix, metadata

def test_create_core_fringe_graph():
    """Test create_core_fringe_graph using a clique and fringe nodes."""
    adj_matrix, metadata = create_test_clique_and_tree()
    
    # Test with dorm_id=1 (clique nodes)
    core_fringe_adj, core_indices, fringe_indices = create_core_fringe_graph(
        adj_matrix, metadata, target_dorm_id=1
    )
    
    # Assertions
    assert len(core_indices) == 4  # 4 nodes in clique
    assert len(fringe_indices) == 3  # 3 fringe nodes
    assert set(core_indices) == {0, 1, 2, 3}  # clique nodes
    assert set(fringe_indices) == {4, 5, 6}   # fringe nodes
    
    # Check that core-fringe adjacency matrix has correct structure
    core_fringe_adj_dense = core_fringe_adj.toarray()
    # Core-core edges should be complete
    assert np.sum(core_fringe_adj_dense[:4, :4]) == 12  # 6 edges * 2 (undirected)
    # Core-fringe edges should exist between node 0 and all fringe nodes
    assert np.sum(core_fringe_adj_dense[:4, 4:]) == 3  # 3 edges (undirected)
    assert np.sum(core_fringe_adj_dense[4:, 4:]) == 0  # no fringe-fringe edges

def test_create_multi_dorm_core_fringe_graph():
    """Test create_multi_dorm_core_fringe_graph using two cliques."""
    adj_matrix, metadata = create_test_two_cliques()
    
    # Test with both dorm_ids
    core_fringe_adj, core_indices, fringe_indices = create_multi_dorm_core_fringe_graph(
        adj_matrix, metadata, target_dorm_ids=[1, 2]
    )
    
    # Assertions
    assert len(core_indices) == 8  # all nodes should be in core
    assert len(fringe_indices) == 0  # no fringe nodes
    
    # Check that core-fringe adjacency matrix has correct structure
    core_fringe_adj_dense = core_fringe_adj.toarray()
    # Should have two complete cliques
    assert np.sum(core_fringe_adj_dense[:4, :4]) == 12  # first clique
    assert np.sum(core_fringe_adj_dense[4:, 4:]) == 12  # second clique
    assert np.sum(core_fringe_adj_dense[:4, 4:]) == 4   # 4 core-fringe edges (undirected)

def test_link_logistic_regression_core_only_auc():
    """Test link_logistic_regression_core_only_auc using two perfectly separable cliques."""
    adj_matrix, metadata = create_test_two_cliques()
    
    # Use first clique as core, second as fringe
    core_indices = np.array([0, 1, 2, 3], dtype=np.int32)
    fringe_indices = np.array([4, 5, 6, 7], dtype=np.int32)
    y_core = metadata[core_indices, 1]  # gender labels for core
    y_fringe = metadata[fringe_indices, 1]  # gender labels for fringe
    
    # Test with 100% of core nodes labeled
    percentages = [1.0]
    lr_kwargs = {'C': 0.1, 'solver': 'liblinear', 'max_iter': 1000}
    auc_dict, betas = link_logistic_regression_core_only_auc(
        adj_matrix=adj_matrix,
        core_indices=core_indices,
        fringe_indices=fringe_indices,
        y_core=y_core,
        y_fringe=y_fringe,
        percentages=percentages,
        lr_kwargs=lr_kwargs,
        seed=42
    )
    
    # Assertions
    assert 1.0 in auc_dict  # check that we have results for 100%
    aucs = auc_dict[1.0]
    assert aucs is not None  # check that we got results
    assert all(auc == 1.0 for auc in aucs if auc is not None)  # perfect separation should give AUC=1.0 