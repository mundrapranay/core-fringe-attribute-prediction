from __future__ import division
# from define_paths import * 
from methods import *
import os
from os import listdir
from os.path import join as path_join
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix
import networkx as nx
from scipy.io import loadmat
from scipy.special import comb
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import random
import math

fb_code_path = '/Users/pranaymundra/Desktop/research_code/core-fringe-attribute-prediction/data/fb100/' ## directory path of raw FB100 .mat files 


def parse_fb100_mat_file(filename):
    """ 
    Parse FB100 .mat files and remove nodes with missing gender or dorm.
    
    Metadata values:
        0 - student/faculty status
        1 - gender
        2 - major
        3 - second major/minor
        4 - dorm/house
        5 - year
        6 - high school
    ** Missing data coded as 0 **
    
    Parameters:
      - filename: path to FB100 .mat file

    Returns:
      (adj_matrix, metadata)
         adj_matrix: cleaned adjacency matrix (only nodes with gender ≠ 0 and dorm ≠ 0)
         metadata: corresponding metadata for remaining nodes
    """
    mat = loadmat(filename)
    error_msg = "%s is not a valid FB100 .mat file. Must contain variable '%s'"
    
    if 'A' not in mat:
        raise ValueError(error_msg % (filename, 'A'))
    adj_matrix = mat['A']
    
    if 'local_info' not in mat:
        raise ValueError(error_msg % (filename, 'local_info'))
    metadata = mat['local_info']
    
    # Remove nodes with missing gender or dorm (both coded as 0)
    gender = metadata[:, 1]
    dorm = metadata[:, 4]
    valid_nodes = np.where((gender != 0) & (dorm != 0))[0]

    metadata = metadata[valid_nodes]
    adj_matrix = csr_matrix(adj_matrix)  # ensure sparse format
    adj_matrix = adj_matrix[valid_nodes][:, valid_nodes]

    return adj_matrix, metadata

def create_core_fringe_graph(adj_matrix, metadata, target_dorm_id):
    """
    Creates a core-fringe graph for a given dorm.
    Core nodes are those whose dorm equals target_dorm_id.
    Fringe nodes are any nodes connected to the core (but not in the core).
    
    Parameters:
      - adj_matrix (scipy.sparse matrix): full graph adjacency matrix.
      - metadata (np.array): metadata for each node.
      - target_dorm_id: dorm id to use for selecting core nodes.
      
    Returns:
      (core_fringe_adj, core_indices, fringe_indices)
    """
    dorms = metadata[:, 4]
    
    # Identify core nodes (by dorm membership)
    core_indices = np.where(dorms == target_dorm_id)[0]
    is_core = np.zeros(adj_matrix.shape[0], dtype=bool)
    is_core[core_indices] = True

    # Ensure CSR format
    adj_matrix = csr_matrix(adj_matrix)

    # Print total number of edges in the original adjacency
    total_edges_original = int(adj_matrix.nnz / 2)
    print(f"Total edges in original adjacency: {total_edges_original}")

    neighbors = adj_matrix[core_indices].nonzero()[1]
    fringe_indices = np.setdiff1d(np.unique(neighbors), core_indices)
    
    # Build mask: keep all core-core and core-fringe edges
    mask = np.zeros_like(adj_matrix.toarray(), dtype=bool)
    # Core-core edges
    for i in core_indices:
        for j in core_indices:
            if adj_matrix[i, j]:
                mask[i, j] = True
                mask[j, i] = True
    # Core-fringe edges
    for i in core_indices:
        for j in fringe_indices:
            if adj_matrix[i, j]:
                mask[i, j] = True
                mask[j, i] = True

    core_fringe_adj = csr_matrix(adj_matrix.multiply(mask))

    # Print total number of edges after core-fringe construction
    total_edges_core_fringe = int(core_fringe_adj.nnz / 2)
    print(f"Total edges after core-fringe construction: {total_edges_core_fringe}")

    # Print number of core-core and core-fringe edges
    num_core_edges = int(np.sum(core_fringe_adj[core_indices][:, core_indices]) / 2)
    core_fringe_edges = int(np.sum(core_fringe_adj[core_indices][:, fringe_indices]) / 2)
    
    print(f"dormID: {target_dorm_id}")
    print(f"Core size: {len(core_indices)}")
    print(f"Number of core-core edges: {num_core_edges}")
    print(f"Number of core-fringe edges: {core_fringe_edges}")

    # After constructing core_fringe_adj
    fringe_adj = core_fringe_adj[fringe_indices, :][:, fringe_indices]
    assert fringe_adj.nnz == 0, "Fringe-fringe edges exist in the core-fringe adjacency matrix!"

    return core_fringe_adj, core_indices, fringe_indices

def create_iid_core_fringe_graph(adj_matrix, k, seed=None, ff=False):
    """
    Creates a core-fringe graph using an IID sample of size k as the core.
    The fringe consists of all nodes connected to the core.
    
    Parameters:
      - adj_matrix (scipy.sparse matrix): full graph adjacency matrix.
      - k (int): number of core nodes to sample.
      - seed (int, optional): random seed.
      
    Returns:
      (core_fringe_adj, core_indices, fringe_indices)
    """
    if seed is not None:
        np.random.seed(seed)
    
    core_indices = np.random.choice(adj_matrix.shape[0], size=k, replace=False)
    is_core = np.zeros(adj_matrix.shape[0], dtype=bool)
    is_core[core_indices] = True

    A = csr_matrix(adj_matrix)
    total_edges_original = int(A.nnz / 2)
    print(f"Total edges in original adjacency: {total_edges_original}")
    neighbors = A[core_indices].nonzero()[1]
    fringe_indices = np.setdiff1d(np.unique(neighbors), core_indices)
    
    fringe_frignge_adj = A[fringe_indices, :][:, fringe_indices]
    # Build a mask that keeps only core-core and core-fringe edges.
    # Create sparse mask matrix
    mask_data = []
    mask_rows = []
    mask_cols = []
    
    # Add core-core edges
    core_core_edges = A[core_indices][:, core_indices]
    for i, j in zip(*np.triu_indices_from(core_core_edges, k=1)):
        mask_rows.extend([core_indices[i], core_indices[j]])
        mask_cols.extend([core_indices[j], core_indices[i]])
        mask_data.extend([1, 1])
    
    # Add core-fringe edges
    core_fringe_edges = A[core_indices][:, fringe_indices]
    for i, j in zip(core_fringe_edges.nonzero()[0], core_fringe_edges.nonzero()[1]):
        mask_rows.extend([core_indices[i], fringe_indices[j]])
        mask_cols.extend([fringe_indices[j], core_indices[i]])
        mask_data.extend([1, 1])

    # Create sparse mask matrix
    mask = csr_matrix((mask_data, (mask_rows, mask_cols)), shape=A.shape)

    core_fringe_adj = A.multiply(mask)

    print(f"IID core")
    print(f"Core size: {len(core_indices)}")
    core_core_edges = int(np.sum(core_fringe_adj[core_indices][:, core_indices]) / 2)
    core_fringe_edges = int(np.sum(core_fringe_adj[core_indices][:, fringe_indices]) / 2)
    print(f"Number of core-core edges: {core_core_edges}")
    print(f"Number of core-fringe edges: {core_fringe_edges}")
    
    # Calculate number of fringe-fringe edges lost
    fringe_fringe_edges = total_edges_original - (core_core_edges + core_fringe_edges)
    print(f"Number of fringe-fringe edges (lost): {fringe_fringe_edges}")

    # After constructing core_fringe_adj
    fringe_adj = core_fringe_adj[fringe_indices, :][:, fringe_indices]
    assert fringe_adj.nnz == 0, "Fringe-fringe edges exist in the core-fringe adjacency matrix!"

    if ff:
        return core_fringe_adj, core_indices, fringe_indices, fringe_frignge_adj
    else:
        return core_fringe_adj, core_indices, fringe_indices


    
def create_multi_dorm_core_fringe_graph(adj_matrix, metadata, target_dorm_ids):
    """
    Creates a core-fringe graph for a given list of dormIDs.
    Core nodes are those whose dorm is in target_dorm_ids.
    Fringe nodes are any nodes that are connected to at least one core node,
    but are not themselves core nodes.
    
    Parameters:
      - adj_matrix (scipy.sparse matrix): full graph adjacency matrix.
      - metadata (np.array): metadata for each node.
      - target_dorm_ids (list or array-like): dorm IDs to be used for the core.
      
    Returns:
      (core_fringe_adj, core_indices, fringe_indices)
         core_fringe_adj: the adjacency matrix after preserving edges only between core-core and core-fringe nodes.
         core_indices: indices (with respect to adj_matrix) of nodes belonging to any dorm in target_dorm_ids.
         fringe_indices: indices (with respect to adj_matrix) of nodes that are connected to the core but not in it.
    """
    dorms = metadata[:, 4]
    # Select core nodes: those whose dorm is in target_dorm_ids.
    core_indices = np.where(np.isin(dorms, target_dorm_ids))[0]
    is_core = np.zeros(adj_matrix.shape[0], dtype=bool)
    is_core[core_indices] = True

    # Ensure CSR format.
    A = csr_matrix(adj_matrix)
    
    # Print total number of edges in the original adjacency
    total_edges_original = int(A.nnz / 2)
    print(f"Total edges in original adjacency: {total_edges_original}")
    # Find all neighbors of the core nodes.
    neighbors = A[core_indices].nonzero()[1]
    # Fringe nodes: those connected to core nodes but not in core.
    fringe_indices = np.setdiff1d(np.unique(neighbors), core_indices)
    
    # Build a mask that keeps only core-core and core-fringe edges.
    # Create sparse mask matrix
    mask_data = []
    mask_rows = []
    mask_cols = []
    
    # Add core-core edges
    core_core_edges = A[core_indices][:, core_indices]
    for i, j in zip(*np.triu_indices_from(core_core_edges, k=1)):
        mask_rows.extend([core_indices[i], core_indices[j]])
        mask_cols.extend([core_indices[j], core_indices[i]])
        mask_data.extend([1, 1])
    
    # Add core-fringe edges
    core_fringe_edges = A[core_indices][:, fringe_indices]
    for i, j in zip(core_fringe_edges.nonzero()[0], core_fringe_edges.nonzero()[1]):
        mask_rows.extend([core_indices[i], fringe_indices[j]])
        mask_cols.extend([fringe_indices[j], core_indices[i]])
        mask_data.extend([1, 1])

    # Create sparse mask matrix
    mask = csr_matrix((mask_data, (mask_rows, mask_cols)), shape=A.shape)

    core_fringe_adj = A.multiply(mask)

    print(f"Multi-dorm core using dormIDs {target_dorm_ids}")
    print(f"Core size: {len(core_indices)}")
    core_core_edges = int(np.sum(core_fringe_adj[core_indices][:, core_indices]) / 2)
    core_fringe_edges = int(np.sum(core_fringe_adj[core_indices][:, fringe_indices]) / 2)
    print(f"Number of core-core edges: {core_core_edges}")
    print(f"Number of core-fringe edges: {core_fringe_edges}")
    
    # Calculate number of fringe-fringe edges lost
    fringe_fringe_edges = total_edges_original - (core_core_edges + core_fringe_edges)
    print(f"Number of fringe-fringe edges (lost): {fringe_fringe_edges}")

    # After constructing core_fringe_adj
    fringe_adj = core_fringe_adj[fringe_indices, :][:, fringe_indices]
    assert fringe_adj.nnz == 0, "Fringe-fringe edges exist in the core-fringe adjacency matrix!"

    return core_fringe_adj, core_indices, fringe_indices

def create_and_save_core_fringe_graph(adj_matrix, metadata, target_dorm_ids):
    core_fringe_adj, core_indices, fringe_indices = create_multi_dorm_core_fringe_graph(adj_matrix, metadata, target_dorm_ids)
    
    save_core_fringe_graph(core_fringe_adj, core_indices, fringe_indices, metadata, output_prefix="Yale_31_32")

def save_core_fringe_graph(core_fringe_adj, core_indices, fringe_indices, metadata, output_prefix="core_fringe"):
    """
    Save the core-fringe adjacency matrix and core indices.
    The adjacency matrix is saved as a dense numpy array.
    
    Parameters:
    - core_fringe_adj: scipy.sparse matrix, the core-fringe adjacency matrix
    - core_indices: np.ndarray, indices of core nodes
    - fringe_indices: np.ndarray, indices of fringe nodes
    - output_prefix: str, prefix for output files
    """
    # Convert sparse matrix to dense numpy array
    adj_matrix = core_fringe_adj.toarray()
    
    # Save adjacency matrix
    np.save(f"{output_prefix}_adj.npy", adj_matrix)
    print(f"Saved adjacency matrix to {output_prefix}_adj.npy")
    
    # Save core indices
    np.save(f"{output_prefix}_core.npy", core_indices)
    print(f"Saved core indices to {output_prefix}_core.npy")
    
    # Save fringe indices
    np.save(f"{output_prefix}_fringe.npy", fringe_indices)
    print(f"Saved fringe indices to {output_prefix}_fringe.npy")

    # Save metadata
    np.save(f"{output_prefix}_metadata.npy", metadata)
    print(f"Saved metadata to {output_prefix}_metadata.npy")

    # Print verification statistics
    core_core_edges = int(np.sum(adj_matrix[np.ix_(core_indices, core_indices)]) / 2)
    core_fringe_edges = int(np.sum(adj_matrix[np.ix_(core_indices, fringe_indices)]) / 2)
    total_edges = core_core_edges + core_fringe_edges
    
    print(f"\nSaved graph statistics:")
    print(f"Total edges: {total_edges}")
    print(f"Core-core edges: {core_core_edges}")
    print(f"Core-fringe edges: {core_fringe_edges}")
    print(f"Sum of core-core and core-fringe: {core_core_edges + core_fringe_edges}")
    
    # Verify no fringe-fringe edges
    fringe_fringe_edges = int(np.sum(adj_matrix[np.ix_(fringe_indices, fringe_indices)]) / 2)
    assert fringe_fringe_edges == 0, "Fringe-fringe edges exist in the adjacency matrix!"

def load_core_fringe_graph(adj_file, core_file, fringe_file, metadata_file):
    """
    Load the core-fringe adjacency matrix and core indices.
    
    Parameters:
    - adj_file: str, path to the .npy file containing the adjacency matrix
    - core_file: str, path to the .npy file containing core indices
    
    Returns:
    - adj_matrix: np.ndarray, the core-fringe adjacency matrix
    - core_indices: np.ndarray, indices of core nodes
    - fringe_indices: np.ndarray, indices of fringe nodes
    """
    # Load adjacency matrix and core indices
    adj_matrix = np.load(adj_file)
    core_indices = np.load(core_file)
    fringe_indices = np.load(fringe_file)
    metadata = np.load(metadata_file)   
    
    # Print verification statistics
    core_core_edges = int(np.sum(adj_matrix[np.ix_(core_indices, core_indices)]) / 2)
    core_fringe_edges = int(np.sum(adj_matrix[np.ix_(core_indices, fringe_indices)]) / 2)
    total_edges = core_core_edges + core_fringe_edges
    fringe_fringe_edges = int(np.sum(adj_matrix[np.ix_(fringe_indices, fringe_indices)]) / 2)
    
    print(f"\nLoaded graph statistics:")
    print(f"Total edges: {total_edges}")
    print(f"Core-core edges: {core_core_edges}")
    print(f"Core-fringe edges: {core_fringe_edges}")
    print(f"Fringe-fringe edges: {fringe_fringe_edges}")
    print(f"Sum of core-core and core-fringe: {core_core_edges + core_fringe_edges}")
    
    return adj_matrix, core_indices, fringe_indices, metadata

def check_loaded_core_fringe_graph(adj_file, core_file, 
                                 expected_core_size, 
                                 expected_core_core_edges, 
                                 expected_core_fringe_edges):   
    """
    Load the saved core-fringe graph and verify its statistics.
    
    Parameters:
    - adj_file: str, path to the .npy file containing the adjacency matrix
    - core_file: str, path to the .npy file containing core indices
    - expected_core_size: int, expected number of core nodes
    - expected_core_core_edges: int, expected number of core-core edges
    - expected_core_fringe_edges: int, expected number of core-fringe edges
    """
    adj_matrix, core_indices, fringe_indices = load_core_fringe_graph(adj_file, core_file)
    
    # Compute statistics
    n_nodes = adj_matrix.shape[0]
    core_core_edges = int(np.sum(adj_matrix[np.ix_(core_indices, core_indices)]) / 2)
    core_fringe_edges = int(np.sum(adj_matrix[np.ix_(core_indices, fringe_indices)]) / 2)
    total_edges = core_core_edges + core_fringe_edges
    fringe_fringe_edges = int(np.sum(adj_matrix[np.ix_(fringe_indices, fringe_indices)]) / 2)
    
    print(f"\nVerification:")
    print(f"Loaded graph: {n_nodes} nodes, {total_edges} edges")
    print(f"Core size: {len(core_indices)}")
    print(f"Number of core-core edges: {core_core_edges}")
    print(f"Number of core-fringe edges: {core_fringe_edges}")
    print(f"Number of fringe-fringe edges: {fringe_fringe_edges}")
    
    # Verify statistics
    assert total_edges == core_core_edges + core_fringe_edges, "Total edges mismatch"
    assert len(core_indices) == expected_core_size, "Core size mismatch"
    assert core_core_edges == expected_core_core_edges, "Core-core edges mismatch"
    assert core_fringe_edges == expected_core_fringe_edges, "Core-fringe edges mismatch"
    assert fringe_fringe_edges == 0, "Fringe-fringe edges should not exist!"
    print("All assertions passed!")

def make_core_fringe():
    file_ext = '.mat'
    for f in listdir(fb_code_path):
        if f.endswith(file_ext):
            print(f)
            adj_matrix, metadata = parse_fb100_mat_file(path_join(fb_code_path, f))
            chosen_dorms_list = [[np.uint(31), np.uint(32)]]
            create_and_save_core_fringe_graph(adj_matrix, metadata, chosen_dorms_list)


def core_fringe_sbm(n_core, n_fringe, p_core_core, p_core_fringe, p_fringe_fringe, seed=42):
    """
    Creates a core-fringe graph using a stochastic block model.
    
    Parameters:
      - n_core (int): number of core nodes
      - n_fringe (int): number of fringe nodes
      - p_core_core (float): probability of edge between core nodes
      - p_core_fringe (float): probability of edge between core and fringe nodes
      - p_fringe_fringe (float): probability of edge between fringe nodes
      - seed (int, optional): random seed
      
    Returns:
      (adj_matrix, core_indices, fringe_indices, metadata)
         adj_matrix: numpy array of the adjacency matrix
         core_indices: array of indices for core nodes
         fringe_indices: array of indices for fringe nodes
         metadata: array containing node attributes (gender in column 1)
    """
    probs = [[p_core_core, p_core_fringe], [p_core_fringe, p_fringe_fringe]]
    total_nodes = [n_core, n_fringe]
    # @ToDo: implement this sbm from scratch
    G = nx.stochastic_block_model(total_nodes, probs, seed=seed)
    
    # Create metadata array with gender information
    # Initialize metadata array with zeros (7 columns to match FB100 format)
    metadata = np.zeros((n_core + n_fringe, 7))
    
    # Set random seed for reproducibility
    np.random.seed(seed)
    
    # Randomly assign genders to core nodes (50% gender 1, 50% gender 2)
    # core_genders = np.random.choice([1, 2], size=n_core, replace=True)
    genders = np.random.choice([1, 2], size=n_core + n_fringe, replace=True)
    
    # Set gender values
    for node in G.nodes():
        # if node < n_core:  # Core node
        #     G.nodes[node]['gender'] = core_genders[node]
        # else:  # Fringe node
        #     G.nodes[node]['gender'] = 2
        G.nodes[node]['gender'] = genders[node]
        metadata[node, 1] = G.nodes[node]['gender']  # Column 1 for gender
    
    # Convert graph to adjacency matrix
    sbm_adj_matrix = nx.to_numpy_array(G)
    # Define core and fringe indices
    core_indices = np.arange(n_core)
    fringe_indices = np.arange(n_core, n_core + n_fringe)

    mask_data = []
    mask_row = []
    mask_col = []
    core_core_edges = sbm_adj_matrix[core_indices, :][:, core_indices]
    core_fringe_edges = sbm_adj_matrix[core_indices, :][:, fringe_indices]
    
    for i, j in zip(*np.triu_indices_from(core_core_edges, k=1)):
        mask_row.extend([core_indices[i], core_indices[j]])
        mask_col.extend([core_indices[j], core_indices[i]])
        mask_data.extend([1,1])

    for i, j in zip(core_fringe_edges.nonzero()[0], core_fringe_edges.nonzero()[1]):
        mask_row.extend([core_indices[i], fringe_indices[j]])
        mask_col.extend([fringe_indices[j], core_indices[i]])
        mask_data.extend([1,1])
    
    mask = csr_matrix((mask_data, (mask_row, mask_col)), shape=sbm_adj_matrix.shape)
    adj_matrix = csr_matrix(sbm_adj_matrix).multiply(mask)
    
    # --- Print expected and observed edge statistics ---
    # Expected edges
    exp_core_core = (n_core * (n_core - 1) / 2) * p_core_core
    exp_core_fringe = n_core * n_fringe * p_core_fringe
    exp_fringe_fringe = (n_fringe * (n_fringe - 1) / 2) * p_fringe_fringe
    # Observed edges (undirected, so divide by 2)
    obs_core_core = int(adj_matrix[core_indices][:, core_indices].sum() / 2)
    obs_core_fringe = int(adj_matrix[core_indices][:, fringe_indices].sum())
    obs_fringe_fringe = int(adj_matrix[fringe_indices][:, fringe_indices].sum() / 2)
    print("--- SBM Block Edge Statistics ---")
    print(f"Expected core-core edges:   {exp_core_core:.1f}")
    print(f"Observed core-core edges:   {obs_core_core}")
    print(f"Expected core-fringe edges: {exp_core_fringe:.1f}")
    print(f"Observed core-fringe edges: {obs_core_fringe}")
    print(f"Expected fringe-fringe edges: {exp_fringe_fringe:.1f}")
    print(f"Observed fringe-fringe edges: {obs_fringe_fringe}")
    print("-------------------------------")

    return adj_matrix, core_indices, fringe_indices, metadata


def sbm_manual_core_fringe(n_core, n_fringe, p_core_core, p_core_fringe, seed=None):
    if seed:
        np.random.seed(seed)
    
    A = np.zeros((n_core + n_fringe, n_core + n_fringe))
    for i in range(n_core):
        for j in range(i+1, n_core):
            A[i, j] = np.random.binomial(1, p_core_core)
            A[j, i] = A[i, j]
    
    for i in range(n_core):
        for j in range(n_core, n_core + n_fringe):
            A[i, j] = np.random.binomial(1, p_core_fringe)
            A[j, i] = A[i, j]
    
    core_indices = np.arange(n_core)
    fringe_indices = np.arange(n_core, n_core + n_fringe)
    metadata = np.zeros((n_core + n_fringe, 7))
    metadata[:, 1] = np.random.choice([1, 2], size=n_core + n_fringe, replace=True)

    # --- Print expected and observed edge statistics ---
    exp_core_core = (n_core * (n_core - 1) / 2) * p_core_core
    exp_core_fringe = n_core * n_fringe * p_core_fringe
    obs_core_core = int(A[core_indices][:, core_indices].sum() / 2)
    obs_core_fringe = int(A[core_indices][:, fringe_indices].sum())
    print(f"Expected core-core edges:   {exp_core_core:.1f}")
    print(f"Observed core-core edges:   {obs_core_core}")
    print(f"Expected core-fringe edges: {exp_core_fringe:.1f}")
    print(f"Observed core-fringe edges: {obs_core_fringe}")

    return csr_matrix(A), core_indices, fringe_indices, metadata
            

def sbm_gender_homophily_adj_and_metadata(n_g1, n_g2, p_in, p_out, seed):
    """
    Generate an SBM with two gender blocks (gender homophily):
    - Half nodes are gender 1, half are gender 2
    - High in-gender (within-block) probability, low out-gender (between-block) probability
    Returns:
      adj_matrix: adjacency matrix (numpy array)
      metadata: metadata array (gender in column 1, 7 columns total)
    """
    import numpy as np
    import networkx as nx
    np.random.seed(seed)
    sizes = [n_g1, n_g2]
    probs = [[p_in, p_out], [p_out, p_in]]
    G = nx.stochastic_block_model(sizes, probs, seed=seed)
    adj_matrix = nx.to_numpy_array(G)
    metadata = np.zeros((n_g1 + n_g2, 7))
    metadata[:n_g1, 1] = 1  # First block: gender 1
    metadata[n_g1:, 1] = 2  # Second block: gender 2
    return adj_matrix, metadata



# if __name__ == "__main__":
    # make_core_fringe()
    # Update these values based on your actual saved files and stats
    # adj_file = "Yale_31_32_adj.npy"
    # core_file = "Yale_31_32_core.npy"
    # fringe_file = "Yale_31_32_fringe.npy"
    # metadata_file = "Yale_31_32_metadata.npy"
    # expected_core_size = 976
    # expected_core_core_edges = 33634
    # expected_core_fringe_edges = 27304

    # check_loaded_core_fringe_graph(
    #     adj_file,
    #     core_file,
    #     expected_core_size,
    #     expected_core_core_edges,
    #     expected_core_fringe_edges
    # )
    # core_indices = np.load(core_file)
    # beta_core_only = link_logistic_regression(adj_file, core_file, fringe_file, metadata_file, core_only=True)
    # beta_core_fringe = link_logistic_regression(adj_file, core_file, fringe_file, metadata_file, core_only=False)
    # print("Correlation:", np.corrcoef(beta_core_only, beta_core_fringe[core_indices])[0, 1])
    # plot_beta_comparison(beta_core_only, beta_core_fringe[core_indices], "Yale_31_32")
    # padded_beta_core_only = np.full_like(beta_core_fringe, np.nan)
    # padded_beta_core_only[core_indices] = beta_core_only
    # plot_beta_comparison(padded_beta_core_only, beta_core_fringe, "Yale_31_32_padded")
    # pipeline()
    # iid_pipeline()
    # core_fringe_sbm(150, 50, 0.8, 0.2, 0.0)
    # adj_matrix, core_indices, fringe_indices, metadata = core_fringe_sbm(1000, 400, 0.15, 0.1, 0.0, seed=123)
    # adj_matrix, core_indices, fringe_indices, metadata = sbm_manual_core_fringe(1000, 400, 0.15, 0.1, seed=123)
    # fringe_inclusion_pipeline_and_plot(n_core=1000, n_fringe=400, p_core_core=0.15, p_core_fringe=0.1, p_fringe_fringe=0.0, n_steps=10, lr_kwargs=None, tag="SBM_fringe_inclusion_0.15_0.1_manual")
    # sbm_pipeline(n_runs=5)
    # finetuneLR()
    # sbm_homophily_sweep_pipeline()