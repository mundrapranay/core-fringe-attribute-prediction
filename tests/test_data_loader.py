import pytest
import numpy as np
import networkx as nx
from scipy.sparse import csr_matrix
import sys
import os
from scipy.io import savemat

# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from code.data_loader import (
    create_core_fringe_graph,
    create_multi_dorm_core_fringe_graph,
    link_logistic_regression_core_only_auc,
    parse_fb100_mat_file,
    link_lr_with_expected_fringe_degree_auc
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

def create_test_mat_file(tmp_path):
    """Creates a temporary .mat file for testing parse_fb100_mat_file."""
    # Create a small test graph
    n = 10
    A = np.zeros((n, n))
    # Add some edges
    A[0:5, 0:5] = 1  # first 5 nodes form a clique
    A[5:, 5:] = 1    # last 5 nodes form a clique
    # Add connections between cliques through nodes that won't be removed
    A[1, 6] = A[6, 1] = 1  # connect through nodes 1 and 6
    A[2, 7] = A[7, 2] = 1  # connect through nodes 2 and 7
    np.fill_diagonal(A, 0)  # remove self-loops
    
    # Create metadata
    local_info = np.zeros((n, 7))
    # Set gender (col 1) and dorm (col 4)
    local_info[0:5, 1] = 1  # first 5 nodes gender=1
    local_info[5:, 1] = 2   # last 5 nodes gender=2
    local_info[0:5, 4] = 1  # first 5 nodes dorm=1
    local_info[5:, 4] = 2   # last 5 nodes dorm=2
    
    # Add some missing data to nodes that aren't critical for connectivity
    local_info[0, 1] = 0  # missing gender
    local_info[5, 4] = 0  # missing dorm
    
    # Save to temporary file
    filepath = tmp_path / "test.mat"
    savemat(str(filepath), {'A': A, 'local_info': local_info})
    return filepath

def test_parse_fb100_mat_file(tmp_path):
    """Test parsing of FB100 .mat files and data cleaning."""
    filepath = create_test_mat_file(tmp_path)
    
    # Parse the file
    adj_matrix, metadata = parse_fb100_mat_file(str(filepath))
    
    # Assertions
    assert adj_matrix.shape[0] == 8  # 2 nodes removed due to missing data
    assert metadata.shape[0] == 8
    assert np.all(metadata[:, 1] != 0)  # no missing gender
    assert np.all(metadata[:, 4] != 0)  # no missing dorm
    
    # Check that the adjacency matrix is symmetric
    assert np.all(adj_matrix.toarray() == adj_matrix.toarray().T)
    
    # Check that the graph is connected
    G = nx.from_scipy_sparse_array(adj_matrix)
    assert nx.is_connected(G)

def test_create_core_fringe_graph_edge_cases():
    """Test create_core_fringe_graph with edge cases."""
    # Create a test graph
    G = nx.Graph()
    G.add_nodes_from(range(5))
    G.add_edges_from([(0,1), (1,2), (2,3), (3,4)])
    adj_matrix = nx.to_scipy_sparse_array(G, format='csr', dtype=np.int32)
    adj_matrix.indices = adj_matrix.indices.astype(np.int32)
    adj_matrix.indptr = adj_matrix.indptr.astype(np.int32)
    
    # Create metadata
    metadata = np.zeros((5, 7))
    metadata[0:2, 4] = 1  # first 2 nodes in dorm 1
    metadata[2:, 4] = 2   # last 3 nodes in dorm 2
    
    # Test with non-existent dorm
    core_fringe_adj, core_indices, fringe_indices = create_core_fringe_graph(
        adj_matrix, metadata, target_dorm_id=3
    )
    assert len(core_indices) == 0
    assert len(fringe_indices) == 0
    
    # Test with isolated core
    metadata[0, 4] = 3  # node 0 in dorm 3
    core_fringe_adj, core_indices, fringe_indices = create_core_fringe_graph(
        adj_matrix, metadata, target_dorm_id=3
    )
    assert len(core_indices) == 1
    assert len(fringe_indices) == 1  # node 1 is fringe

def test_create_multi_dorm_core_fringe_graph_edge_cases():
    """Test create_multi_dorm_core_fringe_graph with edge cases."""
    # Create a test graph
    G = nx.Graph()
    G.add_nodes_from(range(6))
    G.add_edges_from([(0,1), (1,2), (2,3), (3,4), (4,5)])
    adj_matrix = nx.to_scipy_sparse_array(G, format='csr', dtype=np.int32)
    adj_matrix.indices = adj_matrix.indices.astype(np.int32)
    adj_matrix.indptr = adj_matrix.indptr.astype(np.int32)
    
    # Create metadata
    metadata = np.zeros((6, 7))
    metadata[0:2, 4] = 1  # first 2 nodes in dorm 1
    metadata[2:4, 4] = 2  # next 2 nodes in dorm 2
    metadata[4:, 4] = 3   # last 2 nodes in dorm 3
    
    # Test with empty dorm list
    core_fringe_adj, core_indices, fringe_indices = create_multi_dorm_core_fringe_graph(
        adj_matrix, metadata, target_dorm_ids=[]
    )
    assert len(core_indices) == 0
    assert len(fringe_indices) == 0
    
    # Test with non-existent dorm
    core_fringe_adj, core_indices, fringe_indices = create_multi_dorm_core_fringe_graph(
        adj_matrix, metadata, target_dorm_ids=[4]
    )
    assert len(core_indices) == 0
    assert len(fringe_indices) == 0

def test_link_lr_with_expected_fringe_degree_auc():
    """Test link_lr_with_expected_fringe_degree_auc with expected degree imputation."""
    # Create a test graph with two cliques
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
    adj_matrix.indices = adj_matrix.indices.astype(np.int32)
    adj_matrix.indptr = adj_matrix.indptr.astype(np.int32)
    
    # Create metadata
    metadata = np.zeros((8, 7))
    metadata[clique1_nodes, 4] = 1  # dorm_id=1 for core
    metadata[clique2_nodes, 4] = 2  # dorm_id=2 for fringe
    
    # Set gender labels
    metadata[[0, 1], 1] = 1  # first two core nodes gender=1
    metadata[[2, 3], 1] = 2  # last two core nodes gender=2
    metadata[clique2_nodes, 1] = 2  # all fringe nodes gender=2
    
    # Test with 100% of core nodes labeled
    core_indices = np.array(clique1_nodes, dtype=np.int32)
    fringe_indices = np.array(clique2_nodes, dtype=np.int32)
    y_core = metadata[core_indices, 1]
    y_fringe = metadata[fringe_indices, 1]
    
    percentages = [1.0]
    lr_kwargs = {'C': 0.1, 'solver': 'liblinear', 'max_iter': 1000}
    
    auc_dict, betas = link_lr_with_expected_fringe_degree_auc(
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
    assert 1.0 in auc_dict
    aucs = auc_dict[1.0]
    assert aucs is not None
    assert all(auc == 1.0 for auc in aucs if auc is not None)  # perfect separation 