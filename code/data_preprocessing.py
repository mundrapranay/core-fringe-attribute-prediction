from __future__ import division
# from define_paths import * 
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
from methods import *
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

def create_iid_core_fringe_graph(adj_matrix, k, seed=None):
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

    # Build a mask that keeps only core-core and core-fringe edges.
    # Create sparse mask matrix
    mask_data = []
    mask_rows = []
    mask_cols = []
    
    # Add core-core edges
    core_core_edges = A[core_indices][:, core_indices]
    for i, j in zip(core_core_edges.nonzero()[0], core_core_edges.nonzero()[1]):
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
    for i, j in zip(core_core_edges.nonzero()[0], core_core_edges.nonzero()[1]):
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


def link_logistic_regression(adj_file, core_file, fringe_file, metadata_file, core_only=False):
    adj_matrix, core_indices, fringe_indices, metadata = load_core_fringe_graph(adj_file, core_file, fringe_file, metadata_file)

    # Get gender and dorm information
    gender = metadata[:, 1]
    dorm = metadata[:, 4]

    # Create core-only adjacency matrix
    if core_only:
        X_train = adj_matrix[core_indices, :][:, core_indices]
        y_train = gender[core_indices]
        X_test = adj_matrix[fringe_indices, :][:, core_indices]
        print("\n Feature Space (Core-Core only)")
        print(f"X_train shape: {X_train.shape}")
        print(f"y_train shape: {y_train.shape}")
        # print(f"Number of non-zero elements in X_train: {X_train.nnz}")
    else:
        X_train = adj_matrix[core_indices, :]
        y_train = gender[core_indices]
        X_test = adj_matrix[fringe_indices, :]
        print("\n Feature Space (Core-Fringe)")
        print(f"X_train shape: {X_train.shape}")
        print(f"y_train shape: {y_train.shape}")
        # print(f"Number of non-zero elements in X_train: {X_train.nnz}")
    
    # X_test = adj_matrix[fringe_indices, :][:, core_indices]
    y_test = gender[fringe_indices]
    seed = 42
    unique_train_classes = np.unique(y_train)
    print(f"Unique training classes: {unique_train_classes}")
    if unique_train_classes.size < 2:
        print("Not enough unique classes for training. Skipping logistic regression.")
        return
    
    # Train logistic regression model
    lr_kwargs = {'C': 0.1, 'solver': 'liblinear', 'max_iter': 1000}
    model = LogisticRegression(**lr_kwargs, random_state=seed)
    model.fit(X_train, y_train)
    beta = model.coef_.flatten()
    print(f"Number of non-zero coefficients: {np.count_nonzero(beta)}")
    print(f"Mean absolute coefficient: {np.mean(np.abs(beta)):.4f}")
    print(f"Max coefficient: {np.max(np.abs(beta)):.4f}")
    print(f"Min coefficient: {np.min(np.abs(beta)):.4f}")
    print(f"Max coefficient (No-Abs): {np.max(beta):.4f}")
    print(f"Min coefficient (No-Abs): {np.min(beta):.4f}")
    
    # Make predictions on test set
    y_test_pred = model.predict(X_test)
    y_test_scores = model.predict_proba(X_test)
    print(f"Test Accuracy: {accuracy_score(y_test, y_test_pred):.4f}")
    print(f"Test ROC AUC: {roc_auc_score(y_test, y_test_scores[:, 1]):.4f}")
    return beta


def finetuneLR():
    from methods import link_logistic_regression_pipeline
    file_ext = '.mat'
    best_auc = -1
    best_C = None
    best_solver = None
    best_beta = None
    best_acc = None
    best_model_type = None  # core_only or not

    # Try a range of C values (regularization strengths) and solvers
    C_values = [0.001, 0.01, 0.1, 1, 10, 100]
    solvers = ['liblinear', 'lbfgs', 'saga', 'newton-cg', 'sag']
    for f in listdir(fb_code_path):
        if f.endswith(file_ext):
            print(f)
            adj_matrix, metadata = parse_fb100_mat_file(path_join(fb_code_path, f))
            chosen_dorms_list = [np.uint(31), np.uint(32)]
            adj_matrix, core_indices, fringe_indices = create_multi_dorm_core_fringe_graph(adj_matrix, metadata, chosen_dorms_list)
            for core_only in [True, False]:
                for solver in solvers:
                    for C in C_values:
                        lr_kwargs = {'C': C, 'solver': solver, 'max_iter': 1000}
                        try:
                            beta, acc, auc = link_logistic_regression_pipeline(
                                adj_matrix, core_indices, fringe_indices, metadata,
                                core_only=core_only, lr_kwargs=lr_kwargs
                            )
                            print(f"core_only={core_only}, solver={solver}, C={C}, acc={acc:.4f}, auc={auc:.4f}")
                            if auc > best_auc:
                                best_auc = auc
                                best_C = C
                                best_solver = solver
                                best_beta = beta
                                best_acc = acc
                                best_model_type = core_only
                        except Exception as e:
                            print(f"Skipped solver={solver}, C={C} due to error: {e}")

    print(f"Best AUC: {best_auc:.4f} with solver={best_solver}, C={best_C}, core_only={best_model_type}, acc={best_acc:.4f}")
    # Optionally, return or save best_beta, best_C, best_solver, etc.

def pipeline():
    file_ext = '.mat'
    auc_scores = {
        'cc' : [],
        'cf' : [],
        'cfed' : []
    }
    acc_scores = {
        'cc' : [],
        'cf' : [],
        'cfed' : []
    }

    for f in listdir(fb_code_path):
        if f.endswith(file_ext):
            print(f)
            adj_matrix, metadata = parse_fb100_mat_file(path_join(fb_code_path, f))
            chosen_dorms_list = [[np.uint(31), np.uint(32)]]
            adj_matrix, core_indices, fringe_indices = create_multi_dorm_core_fringe_graph(adj_matrix, metadata, chosen_dorms_list)
            percentages = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            lr_kwargs = {'C': 0.01, 'solver': 'newton-cg', 'max_iter': 1000}
            for p in percentages:
                labelled_core_indices = np.random.choice(core_indices, size=int(p * len(core_indices)), replace=False)
                beta_core_only_p, acc_cc, auc_cc = link_logistic_regression_pipeline(adj_matrix, labelled_core_indices, fringe_indices, metadata, core_only=True, lr_kwargs=lr_kwargs)
                beta_core_fringe_p, acc_cf, auc_cf = link_logistic_regression_pipeline(adj_matrix, labelled_core_indices, fringe_indices, metadata, core_only=False, lr_kwargs=lr_kwargs)    
                beta_cfed, acc_cfed, auc_cfed = link_logistic_regression_pipeline(adj_matrix, labelled_core_indices, fringe_indices, metadata, core_only=False, lr_kwargs=lr_kwargs, expected_degree=True)
                # print("Correlation:", np.corrcoef(beta_core_only_p, beta_core_fringe_p[labelled_core_indices])[0, 1])
                padded_beta_core_only = np.full_like(beta_core_fringe_p, np.nan)
                padded_beta_core_only[labelled_core_indices] = beta_core_only_p
                plot_beta_comparison(beta_core_fringe_p, padded_beta_core_only, f"Yale_31_32_pipeline_padded_{p}")
                auc_scores['cc'].append(auc_cc)
                auc_scores['cf'].append(auc_cf)
                auc_scores['cfed'].append(auc_cfed)
                acc_scores['cc'].append(acc_cc)
                acc_scores['cf'].append(acc_cf)
                acc_scores['cfed'].append(acc_cfed)
            # beta_core_only = link_logistic_regression_pipeline(adj_matrix, core_indices, fringe_indices, metadata, core_only=True)
            # beta_core_fringe = link_logistic_regression_pipeline(adj_matrix, core_indices, fringe_indices, metadata, core_only=False)
            # print("Correlation:", np.corrcoef(beta_core_only, beta_core_fringe[core_indices])[0, 1])
            # plot_beta_comparison(beta_core_only, beta_core_fringe[core_indices], "Yale_31_32_pipeline")
    plot_auc(auc_scores, acc_scores, percentages, "Yale_31_32")

def plot_auc(auc_scores, acc_scores, percentages, tag):
    plt.figure(figsize=(10, 5))
    plt.plot(percentages, auc_scores['cc'], label='Core-Core', marker='o')
    plt.plot(percentages, auc_scores['cf'], label='Core-Fringe', marker='o')
    plt.plot(percentages, auc_scores['cfed'], label='Core-Fringe (Expected Degree)', marker='o')
    plt.xlabel('Percentage of Core Nodes Used for Training')
    plt.xticks(percentages)
    plt.ylabel('AUC')
    plt.title(f'AUC Comparison for {tag}')
    plt.legend()
    plt.savefig(f"../figures/{tag}_auc_comparison.png")
    plt.cla()
    plt.clf()

    plt.figure(figsize=(10, 5))
    plt.plot(percentages, acc_scores['cc'], label='Core-Core', marker='o')
    plt.plot(percentages, acc_scores['cf'], label='Core-Fringe', marker='o')
    plt.plot(percentages, acc_scores['cfed'], label='Core-Fringe (Expected Degree)', marker='o')
    plt.xlabel('Percentage of Core Nodes Used for Training')
    plt.xticks(percentages)
    plt.ylabel('Accuracy')
    plt.title(f'Accuracy Comparison for {tag}')
    plt.legend()
    plt.savefig(f"../figures/{tag}_acc_comparison.png")
    plt.close()

def iid_pipeline():
    file_ext = '.mat'
    auc_scores = {
        'cc' : [],
        'cf' : [],
        'cfed' : []
    }
    acc_scores = {  
        'cc' : [],
        'cf' : [],
        'cfed' : []
    }
    
    for f in listdir(fb_code_path):
        if f.endswith(file_ext):
            print(f)
            adj_matrix, metadata = parse_fb100_mat_file(path_join(fb_code_path, f))
            assortativity = nx.degree_assortativity_coefficient(nx.from_numpy_array(adj_matrix))
            print(f"Assortativity: {assortativity}")
            # break
            core_fringe_adj, core_indices, fringe_indices = create_iid_core_fringe_graph(adj_matrix, 975, seed=42)
            percentages = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            lr_kwargs = {'C': 0.01, 'solver': 'newton-cg', 'max_iter': 1000}
            for p in percentages:
                labelled_core_indices = np.random.choice(core_indices, size=int(p * len(core_indices)), replace=False)
                beta_core_only, acc_core_only, auc_core_only = link_logistic_regression_pipeline(core_fringe_adj, labelled_core_indices, fringe_indices, metadata, core_only=True, lr_kwargs=lr_kwargs)
                beta_core_fringe, acc_core_fringe, auc_core_fringe = link_logistic_regression_pipeline(core_fringe_adj, labelled_core_indices, fringe_indices, metadata, core_only=False, lr_kwargs=lr_kwargs)
                beta_cfed, acc_cfed, auc_cfed = link_logistic_regression_pipeline(core_fringe_adj, labelled_core_indices, fringe_indices, metadata, core_only=False, lr_kwargs=lr_kwargs, expected_degree=True, iid_core=True)
                # print("Correlation:", np.corrcoef(beta_core_only, beta_core_fringe[labelled_core_indices])[0, 1])
                padded_beta_core_only = np.full_like(beta_core_fringe, np.nan)
                padded_beta_core_only[labelled_core_indices] = beta_core_only
                plot_beta_comparison(beta_core_fringe, padded_beta_core_only, f"Yale_31_32_iid_pipeline_padded_{p}")
                auc_scores['cc'].append(auc_core_only)
                auc_scores['cf'].append(auc_core_fringe)
                auc_scores['cfed'].append(auc_cfed)
                acc_scores['cc'].append(acc_core_only)
                acc_scores['cf'].append(acc_core_fringe)
                acc_scores['cfed'].append(acc_cfed)
    plot_auc(auc_scores, acc_scores, percentages, f"Yale_31_32_iid_pipeline")
            # plot_beta_comparison(beta_core_fringe, beta_core_only, f"Yale_31_32_iid_pipeline")
# def link_logistic_regression_pipeline(adj_matrix, core_indices, fringe_indices, metadata, core_only=False):
#     # Get gender and dorm information
#     gender = metadata[:, 1]
#     dorm = metadata[:, 4]

#     # Create core-only adjacency matrix
#     if core_only:
#         X_train = adj_matrix[core_indices, :][:, core_indices]
#         y_train = gender[core_indices]
#         X_test = adj_matrix[fringe_indices, :][:, core_indices]
#         print("\n Feature Space (Core-Core only)")
#         print(f"X_train shape: {X_train.shape}")
#         print(f"y_train shape: {y_train.shape}")
#         # print(f"Number of non-zero elements in X_train: {X_train.nnz}")
#     else:
#         X_train = adj_matrix[core_indices, :]
#         y_train = gender[core_indices]
#         X_test = adj_matrix[fringe_indices, :]
#         print("\n Feature Space (Core-Fringe)")
#         print(f"X_train shape: {X_train.shape}")
#         print(f"y_train shape: {y_train.shape}")
#         # print(f"Number of non-zero elements in X_train: {X_train.nnz}")
    
#     # X_test = adj_matrix[fringe_indices, :][:, core_indices]
#     y_test = gender[fringe_indices]
#     seed = 42
#     unique_train_classes = np.unique(y_train)
#     print(f"Unique training classes: {unique_train_classes}")
#     if unique_train_classes.size < 2:
#         print("Not enough unique classes for training. Skipping logistic regression.")
#         return
    
#     # Train logistic regression model
#     lr_kwargs = {'C': 0.1, 'solver': 'liblinear', 'max_iter': 1000}
#     model = LogisticRegression(**lr_kwargs, random_state=seed)
#     model.fit(X_train, y_train)
#     beta = model.coef_.flatten()
#     print(f"Number of non-zero coefficients: {np.count_nonzero(beta)}")
#     print(f"Mean absolute coefficient: {np.mean(np.abs(beta)):.4f}")
#     print(f"Max coefficient: {np.max(np.abs(beta)):.4f}")
#     print(f"Min coefficient: {np.min(np.abs(beta)):.4f}")
#     print(f"Max coefficient (No-Abs): {np.max(beta):.4f}")
#     print(f"Min coefficient (No-Abs): {np.min(beta):.4f}")
    
#     # Make predictions on test set
#     y_test_pred = model.predict(X_test)
#     y_test_scores = model.predict_proba(X_test)
#     print(f"Test Accuracy: {accuracy_score(y_test, y_test_pred):.4f}")
#     print(f"Test ROC AUC: {roc_auc_score(y_test, y_test_scores[:, 1]):.4f}")
#     return beta

def plot_beta_comparison(beta_all, beta_core, tag):
    """
    Generates and saves a scatter plot comparing two β vectors,
    padding the shorter vector with NaNs so both vectors align by index.

    Parameters:
    - beta_all: np.ndarray, coefficients from model trained on all edges
    - beta_core: np.ndarray, coefficients from model trained on core-core edges
    - tag: str, prefix to use in output filenames
    """
    # Determine max length
    max_len = max(beta_all.shape[0], beta_core.shape[0])
    # Pad shorter with NaN
    bx = np.full(max_len, np.nan)
    by = np.full(max_len, np.nan)
    bx[:beta_all.shape[0]] = beta_all
    by[:beta_core.shape[0]] = beta_core

    mask = ~np.isnan(bx) & ~np.isnan(by)
    x_vals = bx[mask]
    y_vals = by[mask]

    # Create XY comparison plot
    fig, ax = plt.subplots(figsize=(6,6), dpi=150)

    # Add hexbin underlay for density
    hb = ax.hexbin(x_vals, y_vals,
                   gridsize=50,
                   cmap='Blues',
                   mincnt=1,
                   alpha=0.4)

    # Overlay the raw points
    ax.scatter(x_vals, y_vals,
               s=15,          # smaller dots
               c='purple',    # override so points show over hexbin
               alpha=0.6,
               label='features')

    # Add identity line y = x
    mn = np.nanmin([x_vals.min(), y_vals.min()]) * 1.1
    mx = np.nanmax([x_vals.max(), y_vals.max()]) * 1.1
    ax.plot([mn, mx], [mn, mx], 'k--', linewidth=1, label='y = x')

    ax.set_xlabel("β_all (all edges)")
    ax.set_ylabel("β_core (core-core edges)")
    ax.set_title(f"β_all vs β_core ({tag})")
    ax.legend(loc='upper left', fontsize='small')
    ax.grid(True, linestyle=':', alpha=0.5)

    fname_xy = f"../figures/{tag}_beta_compare_new_code.png"
    fig.savefig(fname_xy, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved XY comparison plot to {fname_xy}")

if __name__ == "__main__":
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
    iid_pipeline()