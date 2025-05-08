from __future__ import division
# from define_paths import * 
import os
from os import listdir
from os.path import join as path_join
import numpy as np
from scipy.sparse import csr_matrix
import networkx as nx
from scipy.io import loadmat
from scipy.special import comb
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
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

#@ToDO: unit test for the core-fringe edges
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

    # Find all neighbors of core nodes
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
    
    # Print some stats
    num_core_nodes = len(core_indices)
    num_core_edges = int(np.sum(core_fringe_adj[core_indices][:, core_indices]) / 2)
    num_core_fringe_edges = int(np.sum(core_fringe_adj[core_indices][:, fringe_indices]))
    
    print(f"dormID: {target_dorm_id}")
    print(f"Core size: {num_core_nodes}")
    print(f"Number of core-core edges: {num_core_edges}")
    print(f"Number of core-fringe edges: {num_core_fringe_edges}")
    
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
    n = adj_matrix.shape[0]
    core_indices = np.random.choice(n, size=k, replace=False)
    is_core = np.zeros(n, dtype=bool)
    is_core[core_indices] = True
    
    adj_matrix = csr_matrix(adj_matrix)
    neighbors = adj_matrix[core_indices].nonzero()[1]
    fringe_indices = np.setdiff1d(np.unique(neighbors), core_indices)
    
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
    
    num_core_nodes = len(core_indices)
    num_core_edges = int(np.sum(core_fringe_adj[core_indices][:, core_indices]) / 2)
    num_core_fringe_edges = int(np.sum(core_fringe_adj[core_indices][:, fringe_indices]))
    
    print(f"IID Core size: {num_core_nodes}")
    print(f"Number of core-core edges: {num_core_edges}")
    print(f"Number of core-fringe edges: {num_core_fringe_edges}")
    
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
    
    # Find all neighbors of the core nodes.
    neighbors = A[core_indices].nonzero()[1]
    # Fringe nodes: those connected to core nodes but not in core.
    fringe_indices = np.setdiff1d(np.unique(neighbors), core_indices)
    
    # Build a mask that keeps only core-core and core-fringe edges.
    mask = np.zeros_like(A.toarray(), dtype=bool)
    # Mark core-core edges.
    for i in core_indices:
        for j in core_indices:
            if A[i, j]:
                mask[i, j] = True
                mask[j, i] = True
    # Mark core-fringe edges.
    for i in core_indices:
        for j in fringe_indices:
            if A[i, j]:
                mask[i, j] = True
                mask[j, i] = True

    core_fringe_adj = csr_matrix(A.multiply(mask))

    print(f"Multi-dorm core using dormIDs {target_dorm_ids}")
    print(f"Core size: {len(core_indices)}")
    core_core_edges = int(np.sum(core_fringe_adj[core_indices][:, core_indices]) / 2)
    core_fringe_edges = int(np.sum(core_fringe_adj[core_indices][:, fringe_indices]))
    print(f"Number of core-core edges: {core_core_edges}")
    print(f"Number of core-fringe edges: {core_fringe_edges}")
    
    return core_fringe_adj, core_indices, fringe_indices

def prepare_core_fringe_attributes(metadata, core_indices, fringe_indices):
    """
    Separates gender attributes for core and fringe nodes.
    Assumes gender is stored in column 1 of metadata.
    
    Parameters:
      - metadata (np.array): metadata for each node.
      - core_indices (np.array): indices of core nodes.
      - fringe_indices (np.array): indices of fringe nodes.
      
    Returns:
      (gender_core, gender_fringe_true)
    """
    gender = metadata[:, 1]
    gender_core = gender[core_indices]
    gender_fringe_true = gender[fringe_indices]
    print(f"Number of core nodes with known gender: {len(gender_core)}")
    print(f"Number of fringe nodes to predict: {len(gender_fringe_true)}")
    return gender_core, gender_fringe_true

# def link_logistic_regression_core_only(adj_matrix, core_indices, fringe_indices, y_core, y_fringe, percentages, seed=None):
#     """
#     For each percentage in 'percentages', randomly sample that fraction of core nodes (using only core-core connectivity)
#     to train a LINK logistic regression model. Then predict fringe node attributes using only their connectivity to the core.
    
#     If the sampled training set contains only one class, the model training is skipped for that percentage.
    
#     Parameters:
#       - adj_matrix (scipy.sparse matrix): full (or core-fringe) adjacency matrix.
#       - core_indices (np.array): indices for core nodes.
#       - fringe_indices (np.array): indices for fringe nodes.
#       - y_core (np.array): labels for core nodes.
#       - y_fringe (np.array): labels for fringe nodes.
#       - percentages (list or np.array): percentages of labelled core nodes to use.
#       - seed (int, optional): random seed.
      
#     Returns:
#       accuracy_dict: a dictionary mapping each percentage to the fringe prediction accuracy,
#                      or None if training was skipped.
#     """
#     if seed is not None:
#         np.random.seed(seed)
        
#     accuracy_dict = {}
#     # For training, restrict features to core-core connectivity:
#     X_core = adj_matrix[core_indices, :][:, core_indices]
#     # For fringe nodes, use only their connections to core nodes:
#     X_fringe = adj_matrix[fringe_indices, :][:, core_indices]
    
#     num_core = len(core_indices)
    
#     for p in percentages:
#         num_labelled = int(np.ceil(p * num_core))
#         labelled_idx = np.random.choice(np.arange(num_core), size=num_labelled, replace=False)
#         X_train = X_core[labelled_idx]
#         y_train = y_core[labelled_idx]
        
#         # Check that there are at least two classes in the training data.
#         unique_classes = np.unique(y_train)
#         if unique_classes.size < 2:
#             print(f"Skipping training for {p*100:.0f}% labelled core nodes because only one class is present: {unique_classes}")
#             accuracy_dict[p] = None
#             continue
        
#         model = LogisticRegression(solver='liblinear', max_iter=1000)
#         model.fit(X_train, y_train)
        
#         y_pred = model.predict(X_fringe)
#         acc = accuracy_score(y_fringe, y_pred)
#         print(f"Percentage of labelled core nodes: {p*100:.0f}% - Fringe Prediction Accuracy: {acc:.2%}")
#         # print("Classification Report:")
#         # print(classification_report(y_fringe, y_pred))
#         accuracy_dict[p] = acc
        
#     return accuracy_dict

# def estimate_expected_degree_from_core(adj_matrix, core_indices):
#     """
#     Estimate the expected degree for fringe–fringe edges from the core sample.
#     Here we simply take the average core degree.
#     """
#     A = csr_matrix(adj_matrix)
#     core_degrees = np.array(A[core_indices, :].sum(axis=1)).flatten()
#     return core_degrees.mean()

def estimate_expected_degree_from_core(adj_matrix, core_indices, fringe_indices):
    """
    For each fringe node, compute its expected fringe–fringe degree
    as the *average* degree of the core nodes it connects to.
    If a fringe node has no core neighbors, fall back to the global
    average core degree.

    Returns:
      expected_deg_fringe: np.array of shape (len(fringe_indices),)
    """
    A = csr_matrix(adj_matrix)
    # 1) compute degree of each core node
    core_deg = np.array(A[core_indices, :].sum(axis=1)).flatten()
    global_avg = core_deg.mean()

    expected = np.zeros(len(fringe_indices), dtype=float)

    # 2) for each fringe node, find which core nodes it connects to
    #    and average their degrees
    for i, f in enumerate(fringe_indices):
        # get row of adjacency for fringe node f w.r.t. core_indices
        # use A[core_indices, f]
        connections = A[core_indices, f].toarray().flatten()  # 1/0
        neighbor_positions = np.nonzero(connections)[0]
        if neighbor_positions.size > 0:
            expected[i] = core_deg[neighbor_positions].mean()
        else:
            expected[i] = global_avg

    return expected

def link_lr_with_expected_fringe_degree_auc(
    adj_matrix,
    core_indices,
    fringe_indices,
    y_core,
    y_fringe,
    percentages,
    lr_kwargs=None,
    seed=42
):
    """
    For each p in 'percentages':
      1) Sample p·|core| core nodes
      2) Build X_train from core rows (all columns)
      3) Build X_fringe from fringe rows (all columns), but replace columns in 'fringe_indices'
         with the single expected degree value instead of zeros.
      4) Train LR and compute per-class AUC.
    Returns:
      auc_dict mapping p → tuple of class-wise AUCs (or None if skipped).
    """
    if lr_kwargs is None:
        lr_kwargs = {'solver':'liblinear', 'max_iter':1000}

    np.random.seed(seed)
    random.seed(seed)

    A = csr_matrix(adj_matrix)
    A_dense = A.toarray()
    n = A_dense.shape[0]

    # Precompute expected fringe-degree scalar
    # expected_deg = estimate_expected_degree_from_core(adj_matrix, core_indices)
    expected_deg_fringe = estimate_expected_degree_from_core(adj_matrix, core_indices, fringe_indices)
    auc_dict = {}
    betas = {}
    num_core = len(core_indices)

    # Build full feature matrices once
    X_core_full   = A_dense[core_indices, :]    # shape (num_core, n)
    X_fringe_full = A_dense[fringe_indices, :]  # shape (num_fringe, n) (num_fringe, num_cores + num_fringe)
    for p in percentages:
        k = math.ceil(p * num_core)
        sampled_idx = np.random.choice(num_core, size=k, replace=False)
        sampled_core = core_indices[sampled_idx]

        # Training set
        X_train = X_core_full[sampled_idx]  # (k, n)
        y_train = y_core[sampled_idx]

        # Skip if only one class present
        if np.unique(y_train).size < 2:
            print(f"Skipping p={p:.2f}: only one class in sampled core")
            auc_dict[p] = None
            continue
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score

        # Build fringe features and impute expected degree
        Xf = X_fringe_full.copy()  # (num_fringe, n)
        # Replace columns of fringe–fringe positions with expected_deg
        # Find positions in columns corresponding to fringe_indices:
        cols = np.array(fringe_indices)
        # Set entire block Xf[:, cols] = expected_deg
        # Xf[:, cols] = expected_deg
        Xf[np.arange(len(fringe_indices))[:, None], cols[None, :]] = expected_deg_fringe[:, None]
        # print(f'Training Feature Vector: {len(X_train[0])}')
        # print(f'Test Feature Vector: {len(Xf[0])}')

        # Train & evaluate
        clf = LogisticRegression(**lr_kwargs, random_state=seed)
        clf.fit(X_train, y_train)
        betas[p] = clf.coef_.flatten()
        y_scores = clf.predict_proba(Xf)
        classes = clf.classes_
        
        aucs = []
        print(f"Percentage of labelled core nodes: {p*100:.0f}%")
        for idx, c in enumerate(classes):
            y_true_bin = (y_fringe == c).astype(int)
            if np.unique(y_true_bin).size < 2:
                print(f"  Class {c} AUC: N/A (insufficient samples)")
                aucs.append(None)
            else:
                score = roc_auc_score(y_true_bin, y_scores[:, idx])
                print(f"  Class {c} AUC: {score*100:.2f}%")
                aucs.append(score)

        auc_dict[p] = tuple(aucs)

    return auc_dict, betas




def link_logistic_regression_core_only_auc(adj_matrix, core_indices, fringe_indices, y_core, y_fringe, percentages, lr_kwargs=None, seed=42):
    """
    For each percentage in 'percentages', randomly sample that fraction of core nodes (using only core-core connectivity)
    to train a LINK logistic regression model. Then predict fringe node attributes using only their connectivity to the core,
    and compute class-wise AUC values.
    
    If the sampled training set contains only one class, the model training is skipped for that percentage.
    
    Parameters:
      - adj_matrix (scipy.sparse matrix): full (or core-fringe) adjacency matrix.
      - core_indices (np.array): indices for core nodes.
      - fringe_indices (np.array): indices for fringe nodes.
      - y_core (np.array): labels for core nodes.
      - y_fringe (np.array): labels for fringe nodes.
      - percentages (list or np.array): percentages of labelled core nodes to use.
      - seed (int, optional): random seed.
      
    Returns:
      auc_dict: a dictionary mapping each percentage to a tuple of class-wise AUC values,
                e.g. auc_dict[p] = (auc_class1, auc_class2),
                or None if training was skipped for that percentage.
    """
    if seed is not None:
        np.random.seed(seed)
        
    auc_dict = {}
    betas = {}
    # For training, restrict features to core-core connectivity:
    X_core = adj_matrix[core_indices, :][:, core_indices]
    # # For fringe nodes, use only their connections to core nodes:
    X_fringe = adj_matrix[fringe_indices, :][:, core_indices]
    # X_core = adj_matrix[core_indices, :]   # rows are core nodes, columns are all nodes
    # X_fringe = adj_matrix[fringe_indices, :] # rows are fringe nodes, columns are all nodes
    num_core = len(core_indices)
    
    for p in percentages:
        num_labelled = int(np.ceil(p * num_core))
        labelled_idx = np.random.choice(np.arange(num_core), size=num_labelled, replace=False)
        X_train = X_core[labelled_idx]
        y_train = y_core[labelled_idx]
        
        # Check that there are at least two classes in the training data.
        unique_train_classes = np.unique(y_train)
        if unique_train_classes.size < 2:
            print(f"Skipping training for {p*100:.0f}% labelled core nodes because only one class is present: {unique_train_classes}")
            auc_dict[p] = None
            continue
        
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        
        model = LogisticRegression(**lr_kwargs, random_state=seed)
        model.fit(X_train, y_train)
        betas[p] = model.coef_.flatten()
        # Obtain probability estimates on the fringe nodes.
        # The shape of y_scores is (n_samples_fringe, n_classes).
        y_scores = model.predict_proba(X_fringe)
        classes = model.classes_  # The order of classes corresponding to columns in y_scores
        
        # Compute the AUC for each class using one-vs-rest formulation.
        aucs = []
        for idx, c in enumerate(classes):
            # Create binary ground truth: 1 for class c, and 0 for other classes.
            y_true_binary = (y_fringe == c).astype(int)
            # If there are not samples for this class in fringe, skip:
            if np.unique(y_true_binary).size < 2:
                aucs.append(None)
            else:
                auc = roc_auc_score(y_true_binary, y_scores[:, idx])
                aucs.append(auc)
        
        # Print the class-wise AUCs.
        print(f"Percentage of labelled core nodes: {p*100:.0f}%")
        for i, c in enumerate(classes):
            if aucs[i] is not None:
                print(f"  Class {c} AUC: {aucs[i]*100:.2f}%")
            else:
                print(f"  Class {c} AUC: N/A (insufficient samples)")
        
        # Store the AUC tuple for this percentage.
        auc_dict[p] = tuple(aucs)
        
    return auc_dict, betas

def link_logistic_regression_core_only_acc(adj_matrix, core_indices, fringe_indices, y_core, y_fringe, percentages, seed=None):
    """
    For each percentage in 'percentages', randomly sample that fraction of core nodes (using only core-core connectivity)
    to train a LINK logistic regression model. Then predict fringe node attributes using only their connectivity to the core.
    
    If the sampled training set contains only one class, the model training is skipped for that percentage.
    
    Parameters:
      - adj_matrix (scipy.sparse matrix): full (or core-fringe) adjacency matrix.
      - core_indices (np.array): indices for core nodes.
      - fringe_indices (np.array): indices for fringe nodes.
      - y_core (np.array): labels for core nodes.
      - y_fringe (np.array): labels for fringe nodes.
      - percentages (list or np.array): percentages of labelled core nodes to use.
      - seed (int, optional): random seed.
      
    Returns:
      accuracy_dict: a dictionary mapping each percentage to a tuple of class-wise fringe prediction accuracies,
                     e.g. accuracy_dict[p] = (acc_class1, acc_class2),
                     or None if training was skipped for that percentage.
    """
    if seed is not None:
        np.random.seed(seed)
        
    accuracy_dict = {}
    # # For training, restrict features to core-core connectivity:
    # X_core = adj_matrix[core_indices, :][:, core_indices]
    # # For fringe nodes, use only their connections to core nodes:
    # X_fringe = adj_matrix[fringe_indices, :][:, core_indices]
    X_core = adj_matrix[core_indices, :]   # rows are core nodes, columns are all nodes
    X_fringe = adj_matrix[fringe_indices, :] # rows are fringe nodes, columns are all nodes

    num_core = len(core_indices)
    
    for p in percentages:
        num_labelled = int(np.ceil(p * num_core))
        labelled_idx = np.random.choice(np.arange(num_core), size=num_labelled, replace=False)
        X_train = X_core[labelled_idx]
        y_train = y_core[labelled_idx]
        
        # Check that there are at least two classes in the training data.
        unique_train_classes = np.unique(y_train)
        if unique_train_classes.size < 2:
            print(f"Skipping training for {p*100:.0f}% labelled core nodes because only one class is present: {unique_train_classes}")
            accuracy_dict[p] = None
            continue
        
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score
        
        model = LogisticRegression(solver='liblinear', max_iter=1000)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_fringe)
        
        # Compute class-wise accuracy over the fringe nodes.
        # We assume that the fringe labels contain at least the same classes as in training.
        classes = np.sort(np.unique(y_fringe))
        acc_class = []
        for c in classes:
            mask = (y_fringe == c)
            # If there are fringe instances of class c, compute accuracy; otherwise, set to None.
            if np.sum(mask) > 0:
                acc = np.mean(y_pred[mask] == y_fringe[mask])
                acc_class.append(acc)
            else:
                acc_class.append(None)
        
        # Print the class-wise accuracies.
        print(f"Percentage of labelled core nodes: {p*100:.0f}%")
        for i, c in enumerate(classes):
            print(f"  Class {c} accuracy: {acc_class[i]*100:.2f}%")
        
        # Store as a tuple (e.g., (acc_class1, acc_class2))
        accuracy_dict[p] = tuple(acc_class)
        
    return accuracy_dict


# def plot_accuracy_results(accuracy_dict, title=None, save_filename=None):
#     """
#     Plots fringe prediction accuracy as a function of the percentage of labelled core nodes.
    
#     Parameters:
#       - accuracy_dict (dict): maps percentages (as decimals) to accuracy values.
#       - title (str, optional): plot title.
#       - save_filename (str, optional): if provided, saves the plot to this filename.
#     """
#     percentages = sorted(accuracy_dict.keys())
#     accuracies = [accuracy_dict[p] for p in percentages]
#     percentages_percent = [p * 100 for p in percentages]
    
#     plt.figure(figsize=(16, 12))
#     plt.plot(percentages_percent, accuracies, marker='o', linestyle='-', linewidth=2)
#     plt.xlabel("Percentage of Labelled Core Nodes (%)")
#     plt.ylabel("Fringe Prediction Accuracy")
#     if title is not None:
#         plt.title(title)
#     else:
#         plt.title("Fringe Prediction Accuracy vs. Labelled Core Nodes")
#     plt.grid(True)
#     plt.ylim(0, 1)
#     if save_filename is not None:
#         plt.savefig(save_filename)
#         print(f"Plot saved as {save_filename}")
#     plt.cla()
#     plt.clf()
#     plt.close()
#     # plt.show()

def plot_accuracy_results(accuracy_dict, title=None, save_filename=None):
    """
    Plots class-wise fringe prediction accuracy as grouped bars, and overlays
    the overall accuracy (mean of class-wise accuracies) as a line chart with markers,
    as a function of the percentage of labelled core nodes.
    
    Parameters:
      - accuracy_dict (dict): maps percentages (as decimals) to a tuple of accuracy values per class,
                              e.g. {0.1: (acc_class1, acc_class2), 0.2: (acc_class1, acc_class2), ...}.
      - title (str, optional): Plot title.
      - save_filename (str, optional): If provided, saves the plot to this filename.
    """
    # Sort the percentage keys and convert fractions to percentages for the x-axis.
    percentages = sorted(accuracy_dict.keys())
    x_vals = np.array([p * 100 for p in percentages])
    
    # Determine the number of classes from the first valid (non-None) entry.
    for p in percentages:
        if accuracy_dict[p] is not None:
            num_classes = len(accuracy_dict[p])
            break
    else:
        print("No valid accuracy data found.")
        return

    # Build an array for the class-wise accuracies.
    # acc_array will have shape (num_classes, number of percentage groups)
    acc_array = np.zeros((num_classes, len(percentages)))
    for idx, p in enumerate(percentages):
        if accuracy_dict[p] is None:
            # Fill with NaN if no data for this percentage.
            acc_array[:, idx] = np.nan
        else:
            acc_array[:, idx] = np.array(accuracy_dict[p])
    
    # Compute overall accuracy for each percentage as the mean of the class-wise accuracies.
    overall_accuracy = np.nanmean(acc_array, axis=0)
    
    # Setup the bar chart: determine bar width and offsets for the class bars.
    bar_width = 3  # adjust as needed
    total_bars = num_classes  # only class-wise bars
    offsets = np.arange(total_bars) - (total_bars - 1) / 2.0
    
    plt.figure(figsize=(10, 6))
    
    # Plot each class bar.
    for i in range(num_classes):
        x_pos = x_vals + offsets[i] * bar_width
        plt.bar(x_pos, acc_array[i, :], width=bar_width, label=f"Class {i+1}")
    
    # Overlay overall accuracy as a line chart with markers.
    plt.plot(x_vals, overall_accuracy, marker='o', markersize=8, color='black', linestyle='-', label='Overall Accuracy')
    
    plt.xlabel("Percentage of Labelled Core Nodes (%)")
    plt.ylabel("Fringe Prediction AUC")
    if title is not None:
        plt.title(title)
    else:
        plt.title("Fringe Prediction AUC vs. Labelled Core Nodes (Class-wise + Overall)")
    
    # Set x-ticks at the group centers.
    plt.xticks(x_vals, [f"{int(p)}%" for p in x_vals])
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True, axis='y')
    
    if save_filename is not None:
        plt.savefig(save_filename)
        print(f"Plot saved as {save_filename}")
    
    # plt.show()
    plt.cla()
    plt.clf()
    plt.close()


def plot_experiment_results_barline(experiment_results, title_prefix=None, save_filename_prefix=None):
    """
    For each experiment in experiment_results (keyed by a tuple of dormIDs), generate a plot that shows:
      - Class-wise accuracy as grouped bars.
      - Overall accuracy (mean of class-wise accuracies) as a line plot with markers.
      
    For each percentage (x-axis), two groups are plotted:
      - Multi-dorm core experiment (e.g., shifted left by 3 units)
      - Normal (complete graph) experiment (shifted right by 3 units)
      
    Bar widths are 2. The bars for each condition use a similar color palette.
    
    Parameters:
      - experiment_results: dict returned by run_experiments. Keys are tuples (dormIDs),
        each value is a dict with keys 'multi_dorm' and 'normal'. Each of those is a dict mapping 
        percentages (e.g. 0.1, 0.3, …) to a tuple of class-wise accuracies.
      - title_prefix (str, optional): a string to prepend to each plot title.
      - save_filename_prefix (str, optional): if provided, the plot for each experiment is saved
        with a filename composed of this prefix and the dormIDs.
    """
    bar_width = 2
    condition_offset = 2  # offset to separate multi_dorm vs normal conditions

    for dorm_ids, results in experiment_results.items():
        # Extract the condition results:
        multi_dict = results['multi_dorm']
        normal_dict = results['normal']
        
        # Assume both conditions have the same set of percentages.
        percentages = sorted(multi_dict.keys())
        # Convert to percentages on the x-axis.
        x_bases = np.array([p * 100 for p in percentages])
        L = len(x_bases)
        
        # Determine number of classes from the first non-None value in multi_dict.
        num_classes = None
        for p in percentages:
            if multi_dict[p] is not None:
                num_classes = len(multi_dict[p])
                break
        if num_classes is None:
            print(f"No valid data for experiment with dormIDs {dorm_ids}. Skipping plot.")
            continue
        
        # Build arrays for multi_dorm and normal: shape (num_classes, L)
        multi_array = np.empty((num_classes, L))
        normal_array = np.empty((num_classes, L))
        for i, p in enumerate(percentages):
            if multi_dict[p] is None:
                multi_array[:, i] = np.nan
            else:
                multi_array[:, i] = np.array(multi_dict[p])
            if normal_dict[p] is None:
                normal_array[:, i] = np.nan
            else:
                normal_array[:, i] = np.array(normal_dict[p])
        
        # Compute overall accuracy (mean across classes) for each condition
        overall_multi = np.nanmean(multi_array, axis=0)
        overall_normal = np.nanmean(normal_array, axis=0)
        
        # Inner offsets for class bars (centered around zero)
        inner_offsets = (np.arange(num_classes) - (num_classes - 1) / 2.0) * bar_width
        
        plt.figure(figsize=(15, 10))
        
        # For the multi_dorm condition, base x positions shifted left by condition_offset.
        x_base_multi = x_bases - condition_offset
        # For the normal condition, shifted right.
        x_base_normal = x_bases + condition_offset
        
        # Choose color palettes:
        # For example, use a blue palette for multi_dorm and a red/orange palette for normal.
        # We create a list of colors (one for each class) by lightening the base color.
        multi_base_color = np.array([31/255, 119/255, 180/255])  # Matplotlib default blue (normalized)
        normal_base_color = np.array([214/255, 39/255, 40/255])   # Matplotlib default red
        
        # Slightly vary the color for each class by adjusting the brightness.
        multi_colors = [multi_base_color * (0.7 + 0.3 * (i / (num_classes - 1) if num_classes > 1 else 1)) for i in range(num_classes)]
        normal_colors = [normal_base_color * (0.7 + 0.3 * (i / (num_classes - 1) if num_classes > 1 else 1)) for i in range(num_classes)]
        
        # Plot the bars for each class for multi_dorm.
        for i in range(num_classes):
            x_positions = x_base_multi + inner_offsets[i]
            plt.bar(x_positions, multi_array[i, :], width=bar_width, color=multi_colors[i], label=f"Multi-dorm: Class {i+1}" if i == 0 else "")
        
        # Plot the bars for each class for normal.
        for i in range(num_classes):
            x_positions = x_base_normal + inner_offsets[i]
            plt.bar(x_positions, normal_array[i, :], width=bar_width, color=normal_colors[i], label=f"Normal: Class {i+1}" if i == 0 else "")
        
        # Overlay overall accuracy as line plots with markers.
        plt.plot(x_base_multi, overall_multi, marker='o', markersize=8, color='black', linestyle='-', label="Multi-dorm Overall")
        plt.plot(x_base_normal, overall_normal, marker='o', markersize=8, color='gray', linestyle='-', label="Normal Overall")
        
        plt.xlabel("Percentage of Labelled Core Nodes (%)")
        plt.ylabel("Fringe Prediction AUC")
        plt.xticks(x_bases, [f"{int(p)}%" for p in x_bases])
        plt.ylim(0, 1)
        plt.legend()
        plt.grid(True, axis='y')
        
        # Build plot title.
        exp_title = f"Experiment for DormIDs {dorm_ids}"
        if title_prefix:
            exp_title = title_prefix + " - " + exp_title
        plt.title(exp_title)
        
        # Save figure if a prefix is provided.
        if save_filename_prefix:
            fname = f"{save_filename_prefix}_dorms_{'_'.join(map(str, dorm_ids))}.png"
            plt.savefig(fname)
            print(f"Plot saved as {fname}")
        
        plt.show()
        plt.cla()
        plt.close()



# def plot_beta_vectors(betas, tag):
#     """
#     Given a dict mapping percentage -> 1D numpy array of coefficients,
#     generates and saves two plots:
#       1) Scatter plot of each beta vector (β) against feature index.
#       2) Line plot of each beta vector against feature index.
#     The output files will be named:
#       {tag}_beta_scatter.png
#       {tag}_beta_line.png

#     Parameters:
#     - betas: dict[float or str, np.ndarray], each array shape (n_features,)
#     - tag: str, prefix to use in output filenames
#     """
#     # Validate and get feature count
#     lengths = [beta.shape[0] for beta in betas.values()]
#     if len(set(lengths)) != 1:
#         raise ValueError("All beta vectors must have the same length.")
#     n_features = lengths[0]

#     # 1) Scatter plot
#     plt.figure()
#     for p, beta in betas.items():
#         plt.scatter(range(n_features), beta, label=str(p))
#     plt.xlabel("Feature index")
#     plt.ylabel("Coefficient (β)")
#     plt.title(f"Scatter of beta vectors ({tag})")
#     plt.legend()
#     fname_scatter = f"../figures/{tag}_beta_scatter.png"
#     plt.savefig(fname_scatter)
#     plt.close()

#     # 2) Line plot
#     plt.figure()
#     for p, beta in betas.items():
#         plt.plot(range(n_features), beta, label=str(p))
#     plt.xlabel("Feature index")
#     plt.ylabel("Coefficient (β)")
#     plt.title(f"Beta vs. feature index ({tag})")
#     plt.legend()
#     fname_line = f"../figures/{tag}_beta_line.png"
#     plt.savefig(fname_line)
#     plt.close()

#     print(f"Saved scatter plot to {fname_scatter}")
#     print(f"Saved line plot    to {fname_line}")

def plot_beta_vectors(betas, tag):
    """
    Given a dict mapping percentage -> 1D numpy array of coefficients,
    generates and saves **two separate** plots for each key:
      1) Scatter plot of β vs. feature index
      2) Line plot of β vs. feature index
    Filenames and titles include the percentage.

    Parameters:
    - betas: dict[float or str, np.ndarray], each array shape (n_features,)
    - tag: str, prefix to use in output filenames
    """
    # Validate all β-vectors have the same length
    lengths = [beta.shape[0] for beta in betas.values()]
    if len(set(lengths)) != 1:
        raise ValueError("All beta vectors must have the same length.")

    for p, beta in betas.items():
        # Make a filesystem-safe percentage string
        perc_str = str(p).replace('.', 'p')
        n_features = beta.shape[0]

        # 1) Scatter plot (feature index vs. coefficient)
        plt.figure()
        plt.scatter(range(n_features), beta)
        plt.xlabel("Feature index")
        plt.ylabel("Coefficient (β)")
        plt.title(f"Scatter of β for p={p} ({tag})")
        fname_scatter = f"../figures/{tag}_{perc_str}_beta_scatter.png"
        plt.savefig(fname_scatter)
        plt.close()
        print(f"Saved scatter plot to {fname_scatter}")

        # 2) Line plot (feature index vs. coefficient)
        plt.figure()
        plt.plot(range(n_features), beta)
        plt.xlabel("Feature index")
        plt.ylabel("Coefficient (β)")
        plt.title(f"Line plot of β for p={p} ({tag})")
        fname_line = f"../figures/{tag}_{perc_str}_beta_line.png"
        plt.savefig(fname_line)
        plt.close()
        print(f"Saved line plot to {fname_line}")



# def plot_beta_comparison(betas_all, betas_core, tag):
#     """
#     Given two dicts mapping percentage -> 1D numpy array of coefficients,
#     generates and saves a scatter plot of β_all vs. β_core for each percentage,
#     padding the shorter vector with NaNs so both vectors align by index.

#     Parameters:
#     - betas_all: dict[float or str, np.ndarray], coefficients from model trained on all edges
#     - betas_core: dict[float or str, np.ndarray], coefficients from model trained on core-core edges
#     - tag: str, prefix to use in output filenames
#     """
#     # Ensure both dicts have the same keys
#     if set(betas_all.keys()) != set(betas_core.keys()):
#         raise ValueError("The two beta dicts must have the same percentage keys.")

#     for p in betas_all:
#         beta_x = betas_all[p]
#         beta_y = betas_core[p]
#         # Determine max length
#         max_len = max(beta_x.shape[0], beta_y.shape[0])
#         # Pad shorter with NaN
#         bx = np.full(max_len, np.nan)
#         by = np.full(max_len, np.nan)
#         bx[:beta_x.shape[0]] = beta_x
#         by[:beta_y.shape[0]] = beta_y

#         # Make a filesystem-safe percentage string
#         perc_str = str(p).replace('.', 'p')

#         # Scatter plot of β_all (x) vs β_core (y)
#         plt.figure()
#         plt.scatter(bx, by)
#         plt.xlabel("β_all (all edges)")
#         plt.ylabel("β_core (core-core edges)")
#         plt.title(f"β_all vs β_core for p={p} ({tag})")
#         fname = f"../figures/{tag}_{perc_str}_beta_compare.png"
#         plt.savefig(fname)
#         plt.close()
#         print(f"Saved comparison plot to {fname}")



# def plot_beta_comparison(betas_all, betas_core, tag):
#     """
#     Given two dicts mapping percentage -> 1D numpy array of coefficients,
#     generates and saves a scatter plot comparing two β vectors for each percentage,
#     padding the shorter vector with NaNs so both vectors align by index,
#     and colors β_all in red and β_core in blue when plotted against feature index.

#     Parameters:
#     - betas_all: dict[float or str, np.ndarray], coefficients from model trained on all edges
#     - betas_core: dict[float or str, np.ndarray], coefficients from model trained on core-core edges
#     - tag: str, prefix to use in output filenames
#     """
#     # Ensure both dicts have the same keys
#     if set(betas_all.keys()) != set(betas_core.keys()):
#         raise ValueError("The two beta dicts must have the same percentage keys.")

#     for p in betas_all:
#         beta_x = betas_all[p]
#         beta_y = betas_core[p]
#         # Determine max length
#         max_len = max(beta_x.shape[0], beta_y.shape[0])
#         # Pad shorter with NaN
#         bx = np.full(max_len, np.nan)
#         by = np.full(max_len, np.nan)
#         bx[:beta_x.shape[0]] = beta_x
#         by[:beta_y.shape[0]] = beta_y

#         # Make a filesystem-safe percentage string
#         perc_str = str(p).replace('.', 'p')

#         # 1) Scatter plot: β_all vs feature index (red) and β_core vs feature index (blue)
#         plt.figure()
#         plt.scatter(range(max_len), bx, color='blue', label='β_all')  # β_all in blue
#         plt.scatter(range(max_len), by, color='red', label='β_core')  # β_core in red
#         plt.xlabel("Feature index")
#         plt.ylabel("Coefficient (β)")
#         plt.title(f"β_all (red) vs β_core (blue) for p={p} ({tag})")
#         plt.legend()
#         fname_idx = f"../figures/{tag}_{perc_str}_beta_compare_index.png"
#         plt.savefig(fname_idx)
#         plt.close()
#         print(f"Saved index comparison plot to {fname_idx}")

#         # 2) Scatter plot of β_all (x) vs β_core (y), colored by comparison
#         plt.figure()
#         # Mask out NaNs
#         mask = ~np.isnan(bx) & ~np.isnan(by)
#         x_vals = bx[mask]
#         y_vals = by[mask]
#         # Points where β_all >= β_core in blue, else red
#         ge_mask = x_vals >= y_vals
#         lt_mask = x_vals < y_vals
#         plt.scatter(x_vals[ge_mask], y_vals[ge_mask], color='blue', label='β_all ≥ β_core')
#         plt.scatter(x_vals[lt_mask], y_vals[lt_mask], color='red', label='β_all < β_core')
#         plt.xlabel("β_all (all edges)")
#         plt.ylabel("β_core (core-core edges)")
#         plt.title(f"β_all vs β_core for p={p} ({tag})")
#         plt.legend()
#         fname_xy = f"../figures/{tag}_{perc_str}_beta_compare_xy.png"
#         plt.savefig(fname_xy)
#         plt.close()
#         print(f"Saved XY comparison plot to {fname_xy}")



def plot_beta_comparison(betas_all, betas_core, tag):
    """
    Given two dicts mapping percentage -> 1D numpy array of coefficients,
    generates and saves a scatter plot comparing two β vectors for each percentage,
    padding the shorter vector with NaNs so both vectors align by index,
    and colors β_all in red and β_core in blue when plotted against feature index.

    Parameters:
    - betas_all: dict[float or str, np.ndarray], coefficients from model trained on all edges
    - betas_core: dict[float or str, np.ndarray], coefficients from model trained on core-core edges
    - tag: str, prefix to use in output filenames
    """
    # Ensure both dicts have the same keys
    if set(betas_all.keys()) != set(betas_core.keys()):
        raise ValueError("The two beta dicts must have the same percentage keys.")

    for p in betas_all:
        beta_x = betas_all[p]
        beta_y = betas_core[p]
        # Determine max length
        max_len = max(beta_x.shape[0], beta_y.shape[0])
        # Pad shorter with NaN
        bx = np.full(max_len, np.nan)
        by = np.full(max_len, np.nan)
        bx[:beta_x.shape[0]] = beta_x
        by[:beta_y.shape[0]] = beta_y

        # Make a filesystem-safe percentage string
        perc_str = str(p).replace('.', 'p')

        # 1) Scatter plot: β_all vs feature index (red) and β_core vs feature index (blue)
        # plt.figure()
        # plt.scatter(range(max_len), bx, color='blue', label='β_all')  # β_all in blue
        # plt.scatter(range(max_len), by, color='red', label='β_core', alpha=0.5)  # β_core in red
        # plt.xlabel("Feature index")
        # plt.ylabel("Coefficient (β)")
        # plt.title(f"β_all (red) vs β_core (blue) for p={p} ({tag})")
        # plt.legend()
        # fname_idx = f"../figures/{tag}_{perc_str}_beta_compare_index.png"
        # plt.savefig(fname_idx)
        # plt.close()
        # print(f"Saved index comparison plot to {fname_idx}")

        # 2) Scatter plot of β_all (x) vs β_core (y)
        # plt.figure()
        # # Mask out NaNs
        # mask = ~np.isnan(bx) & ~np.isnan(by)
        # x_vals = bx[mask]
        # y_vals = by[mask]
        # # Plot points colored: blue for β_all, red for β_core
        # plt.scatter(x_vals, y_vals, color='blue', label='β_all')
        # plt.scatter(x_vals, y_vals, color='red', label='β_core', alpha=0.5)
        # plt.xlabel("β_all (all edges)")
        # plt.ylabel("β_core (core-core edges)")
        # plt.title(f"β_all vs β_core for p={p} ({tag})")
        # plt.legend()
        # fname_xy = f"../figures/{tag}_{perc_str}_beta_compare_xy.png"
        # plt.savefig(fname_xy)
        # plt.close()
        # print(f"Saved XY comparison plot to {fname_xy}")

        mask = ~np.isnan(bx) & ~np.isnan(by)
        x_vals = bx[mask]
        y_vals = by[mask]

        # --- 2) Improved XY comparison plot ---
        fig, ax = plt.subplots(figsize=(6,6), dpi=150)

        # (a) optional hexbin underlay for density
        hb = ax.hexbin(x_vals, y_vals,
                       gridsize=50,
                       cmap='Blues',
                       mincnt=1,
                       alpha=0.4)

        # (b) overlay the raw points with small, semi-transparent markers
        ax.scatter(x_vals, y_vals,
                   s=15,          # smaller dots
                   c='purple',    # override so points show over hexbin
                   alpha=0.6,
                   label='features')

        # (c) identity line y = x
        mn = np.nanmin([x_vals.min(), y_vals.min()]) * 1.1
        mx = np.nanmax([x_vals.max(), y_vals.max()]) * 1.1
        ax.plot([mn, mx], [mn, mx], 'k--', linewidth=1, label='y = x')

        ax.set_xlabel("β_all (all edges)")
        ax.set_ylabel("β_core (core-core edges)")
        ax.set_title(f"β_all vs β_core for p={p} ({tag})")
        ax.legend(loc='upper left', fontsize='small')
        ax.grid(True, linestyle=':', alpha=0.5)

        fname_xy = f"../figures/{tag}_{perc_str}_beta_compare_all.png"
        fig.savefig(fname_xy, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved XY comparison plot to {fname_xy}")


# === Main Data Loader / Experiment Runner ===

def dorm_core_vs_iid_core():
    """
    For each FB100 .mat file in fb_code_path:
      - Load and parse the file, removing nodes with missing gender or dorm.
      - Print file-level statistics:
          * Number of nodes
          * Number of edges
          * Ratio of males to females
      - For each unique dorm in the file:
          * Run two experiments: one using dorm-based core and one using IID-based core.
          * For each, run the LINK LR model over various percentages of labelled core nodes.
          * Compute dorm statistics (core size, fringe size, number of core edges, core-fringe edges, core gender ratio).
          * Plot the accuracy results with these stats in the title and save the plot.
    """
    
    file_ext = '.mat'
    
    for f in listdir(fb_code_path):
        if f.endswith(file_ext):
            tag = f.replace(file_ext, '')
            print("\n===================================")
            print(f"Processing file: {tag}")
            input_file = path_join(fb_code_path, f)
            adj_matrix, metadata = parse_fb100_mat_file(input_file)
            # File-level statistics
            num_nodes = metadata.shape[0]
            num_edges = int(adj_matrix.nnz / 2)
            genders = metadata[:, 1]
            num_males = np.sum(genders == 1)
            num_females = np.sum(genders == 2)
            gender_ratio = num_males / num_females if num_females > 0 else float('inf')
            print("File-level statistics:")
            print(f"  Number of nodes after parsing: {num_nodes}")
            print(f"  Number of edges: {num_edges}")
            print(f"  Gender ratio (G1:G2): {gender_ratio}")
            
            # Process each dorm
            dorm_ids = np.unique(metadata[:, 4])
            # unique, counts = np.unique(dorm_ids, return_counts=True)

            # print(np.asarray((unique, counts)).T)
            print(f"Number of Dorms: {len(dorm_ids)}")
            dormDict = {
                'Yale4': [31, 41],
                'MIT8' : [284, 236]
            }
            # break
            for dorm in dormDict[tag]:
                print("\n-----------------------------------")
                print(f"Processing dorm: {dorm}")
                percentages = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.0]
                
                # Experiment 1: Dorm-based core
                core_fringe_adj, core_indices, fringe_indices = create_core_fringe_graph(adj_matrix, metadata, np.uint16(dorm))
                core_gender, fringe_gender = prepare_core_fringe_attributes(metadata, core_indices, fringe_indices)
                print("Running LINK LR for dorm-based core...")
                accuracy_results_dorm = link_logistic_regression_core_only_acc(
                    adj_matrix=core_fringe_adj,
                    core_indices=core_indices,
                    fringe_indices=fringe_indices,
                    y_core=core_gender,
                    y_fringe=fringe_gender,
                    percentages=percentages,
                    seed=42
                )
                dorm_core_size = len(core_indices)
                dorm_fringe_size = len(fringe_indices)
                num_core_edges = int(np.sum(core_fringe_adj[core_indices][:, core_indices]) / 2)
                num_core_fringe_edges = int(np.sum(core_fringe_adj[core_indices][:, fringe_indices]))
                core_males = np.sum(core_gender == 1)
                core_females = np.sum(core_gender == 2)
                core_gender_ratio = core_males / core_females if core_females > 0 else float('inf')

                fringe_males = np.sum(fringe_gender == 1)
                fringe_females = np.sum(fringe_gender == 2)
                fringe_gender_ratio = fringe_males / fringe_females if fringe_females > 0 else float('inf')
                
                plot_title_dorm = (
                    f"Dorm-based core (Dorm {dorm}) | "
                    f"Core size: {dorm_core_size}, Fringe size: {dorm_fringe_size} | "
                    f"Core edges: {num_core_edges}, Core-Fringe edges: {num_core_fringe_edges}\n"
                    f"Gender ratio (C1:C2): Core: {core_gender_ratio:.2f} | Fringe: {fringe_gender_ratio:.2f}"
                )
                plot_filename_dorm = f"../figures/core_{dorm}_dorm_{tag}_acc_new.png"
                plot_accuracy_results(accuracy_dict=accuracy_results_dorm,
                                      title=plot_title_dorm,
                                      save_filename=plot_filename_dorm)
                
                # Experiment 2: IID-based core
                core_fringe_adj_iid, core_indices_iid, fringe_indices_iid = create_iid_core_fringe_graph(adj_matrix, k=dorm_core_size, seed=42)
                core_gender_iid, fringe_gender_iid = prepare_core_fringe_attributes(metadata, core_indices_iid, fringe_indices_iid)
                print("Running LINK LR for IID-based core...")
                accuracy_results_iid = link_logistic_regression_core_only_acc(
                    adj_matrix=core_fringe_adj_iid,
                    core_indices=core_indices_iid,
                    fringe_indices=fringe_indices_iid,
                    y_core=core_gender_iid,
                    y_fringe=fringe_gender_iid,
                    percentages=percentages,
                    seed=42
                )
                iid_core_size = len(core_indices_iid)
                iid_fringe_size = len(fringe_indices_iid)
                num_core_edges_iid = int(np.sum(core_fringe_adj_iid[core_indices_iid][:, core_indices_iid]) / 2)
                num_core_fringe_edges_iid = int(np.sum(core_fringe_adj_iid[core_indices_iid][:, fringe_indices_iid]))
                core_males_iid = np.sum(core_gender_iid == 1)
                core_females_iid = np.sum(core_gender_iid == 2)
                core_gender_ratio_iid = core_males_iid / core_females_iid if core_females_iid > 0 else float('inf')
                
                fringe_males = np.sum(fringe_gender_iid == 1)
                fringe_females = np.sum(fringe_gender_iid == 2)
                fringe_gender_ratio = fringe_males / fringe_females if fringe_females > 0 else float('inf')

                plot_title_iid = (
                    f"IID-based core | "
                    f"Core size: {iid_core_size}, Fringe size: {iid_fringe_size} | "
                    f"Core edges: {num_core_edges_iid}, Core-Fringe edges: {num_core_fringe_edges_iid}\n"
                    f"Gender Ratio (C1:C2): Core: {core_gender_ratio_iid:.2f} | Fringe: {fringe_gender_ratio:.2f}"
                )
                plot_filename_iid = f"../figures/core_{dorm}_IID_{tag}_auc.png"
                plot_accuracy_results(accuracy_dict=accuracy_results_iid,
                                      title=plot_title_iid,
                                      save_filename=plot_filename_iid)



# -------------------------------------------------------------------
# 4. Experiment function using multi-dorm core vs. normal graph (complete graph)
# -------------------------------------------------------------------

def perform_experiment_for_multi_dorm_core(adj_matrix, metadata, dorm_ids, percentages, seed=None):
    """
    For a given list of dormIDs (dorm_ids), perform two experiments:
      (a) Multi-dorm core experiment: build a core-fringe graph with core being the union of nodes from the given dorms.
      (b) Normal graph experiment: randomly sample from the entire graph the same number of nodes as in (a) 
          and use the complete graph (with all edges) for training and prediction.
          (Here, training features are the full rows of the training set, and fringe features are the full rows for 
           the complement of the training nodes.)
    
    Then, run the LINK LR model on both cases and return the performance results along with statistics strings.
    
    Returns:
      results: A dictionary with keys:
          'multi_dorm': results from the multi-dorm core experiment.
          'normal': results from the normal graph experiment.
          'sizes': a tuple (multi_dorm_core_size, multi_dorm_fringe_size, normal_fringe_size)
          'stats': a string summarizing multi-dorm core statistics:
                   "Multi Dorm-based core (Dorms ...)| Core size: ..., Fringe size: ... | 
                    Core edges: ..., Core-Fringe edges: ...
                    Gender ratio (C1:C2): Core: ... | Fringe: ..."
          'normal_stats': a similar string for the random sample normal graph.
    """
    # ---------------------
    # (a) Multi-dorm core experiment.
    multi_dorm_graph, multi_dorm_core_indices, multi_dorm_fringe_indices = create_multi_dorm_core_fringe_graph(adj_matrix, metadata, dorm_ids)
    multi_gender_core, multi_gender_fringe = prepare_core_fringe_attributes(metadata, multi_dorm_core_indices, multi_dorm_fringe_indices)
    print(f"=== Multi-Dorm Core Experiment using dormIDs: {dorm_ids} ===")
    print(f"Multi-dorm core size: {len(multi_dorm_core_indices)}")
    
    results_multi = link_logistic_regression_core_only_auc(
        adj_matrix=multi_dorm_graph,
        core_indices=multi_dorm_core_indices,
        fringe_indices=multi_dorm_fringe_indices,
        y_core=multi_gender_core,
        y_fringe=multi_gender_fringe,
        percentages=percentages,
        seed=seed
    )
    
    # Compute multi-dorm core statistics.
    core_size = len(multi_dorm_core_indices)
    fringe_size = len(multi_dorm_fringe_indices)
    num_core_edges = int(np.sum(multi_dorm_graph[multi_dorm_core_indices][:, multi_dorm_core_indices]) / 2)
    num_core_fringe_edges = int(np.sum(multi_dorm_graph[multi_dorm_core_indices][:, multi_dorm_fringe_indices]))
    
    # Compute gender ratios for multi-dorm core.
    core_males = np.sum(multi_gender_core == 1)
    core_females = np.sum(multi_gender_core == 2)
    core_gender_ratio = core_males / core_females if core_females > 0 else float('inf')
    
    fringe_males = np.sum(multi_gender_fringe == 1)
    fringe_females = np.sum(multi_gender_fringe == 2)
    fringe_gender_ratio = fringe_males / fringe_females if fringe_females > 0 else float('inf')
    
    stats_string = (f"Multi Dorm-based core (Dorms {','.join(map(str, dorm_ids))}) | "
                    f"Core size: {core_size}, Fringe size: {fringe_size} | "
                    f"Core edges: {num_core_edges}, Core-Fringe edges: {num_core_fringe_edges}\n"
                    f"Gender ratio (C1:C2): Core: {core_gender_ratio:.2f} | Fringe: {fringe_gender_ratio:.2f}")
    
    # ---------------------
    # (b) Normal graph experiment using the complete graph.
    n = adj_matrix.shape[0]
    k = core_size  # sample same number of nodes as the multi-dorm core size.
    if seed is not None:
        np.random.seed(seed)
    random_core_indices = np.random.choice(np.arange(n), size=k, replace=False)
    # Define fringe as all other nodes (the complement).
    all_nodes = np.arange(n)
    random_fringe_indices = np.setdiff1d(all_nodes, random_core_indices)
    
    # Extract gender attributes for the normal experiment.
    random_gender_core = metadata[random_core_indices, 1]
    random_gender_fringe = metadata[random_fringe_indices, 1]
    
    # Compute normal experiment statistics.
    normal_core_edges = int(np.sum(adj_matrix[random_core_indices, :][:, random_core_indices]) / 2)
    normal_core_fringe_edges = int(np.sum(adj_matrix[random_core_indices, :][:, random_fringe_indices]))
    normal_core_size = len(random_core_indices)
    normal_fringe_size = len(random_fringe_indices)
    
    core_males_normal = np.sum(random_gender_core == 1)
    core_females_normal = np.sum(random_gender_core == 2)
    normal_core_gender_ratio = core_males_normal / core_females_normal if core_females_normal > 0 else float('inf')
    
    fringe_males_normal = np.sum(random_gender_fringe == 1)
    fringe_females_normal = np.sum(random_gender_fringe == 2)
    normal_fringe_gender_ratio = fringe_males_normal / fringe_females_normal if fringe_females_normal > 0 else float('inf')
    
    normal_stats_string = (f"Normal graph (random sample) | "
                           f"Core size: {normal_core_size}, Fringe size: {normal_fringe_size} | "
                           f"Core edges: {normal_core_edges}, Core-Fringe edges: {normal_core_fringe_edges}\n"
                           f"Gender ratio (C1:C2): Core: {normal_core_gender_ratio:.2f} | Fringe: {normal_fringe_gender_ratio:.2f}")
    
    print(f"Normal graph experiment (complete graph):")
    print(f"  Training (random core) size: {normal_core_size}")
    print(f"  Fringe (complement) size: {normal_fringe_size}")
    
    results_normal = link_logistic_regression_core_only_auc(
        adj_matrix=adj_matrix,
        core_indices=random_core_indices,
        fringe_indices=random_fringe_indices,
        y_core=random_gender_core,
        y_fringe=random_gender_fringe,
        percentages=percentages,
        seed=seed
    )
    
    sizes = (core_size, fringe_size, normal_fringe_size)
    return {'multi_dorm': results_multi, 'normal': results_normal, 'sizes': sizes, 'stats': stats_string, 'normal_stats': normal_stats_string}


def run_experiments(adj_matrix, metadata, chosen_dorms_list, percentages, seed=None):
    """
    Run the experiment for each set of dormIDs in chosen_dorms_list.
    For each set, compare the multi-dorm core experiment with the normal graph experiment.
    """
    experiment_results = {}
    for dorm_ids in chosen_dorms_list:
        print("\n===================================")
        print(f"Running experiment for dormIDs: {dorm_ids}")
        results = perform_experiment_for_multi_dorm_core(adj_matrix, metadata, dorm_ids, percentages, seed=seed)
        experiment_results[tuple(dorm_ids)] = results
        
        core_size, fringe_size, normal_fringe_size = results['sizes']
        print(f"Multi-dorm core size: {core_size}, Multi-dorm fringe size: {fringe_size}")
        print(f"Normal graph fringe size: {normal_fringe_size}")
        print("Multi-dorm core results:")
        print(results['multi_dorm'])
        print("Normal graph results:")
        print(results['normal'])
        print("Multi-dorm core stats:")
        print(results['stats'])
        print("Normal graph stats:")
        print(results['normal_stats'])
        
    return experiment_results



if __name__ == '__main__':
    # data_loader()
    file_ext = '.mat'
    
    for f in listdir(fb_code_path):
        if f.endswith(file_ext):
            tag = f.replace(file_ext, '')
            print("\n===================================")
            print(f"Processing file: {tag}")
            input_file = path_join(fb_code_path, f)
            adj_matrix, metadata = parse_fb100_mat_file(input_file)
            chosen_dorms_list = [[np.uint(31), np.uint(32)]]
            percentages = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.0]
            # Run experiments with a fixed seed for reproducibility.
            experiment_results = run_experiments(adj_matrix, metadata, chosen_dorms_list, percentages, seed=42)
            plot_experiment_results_barline(experiment_results, title_prefix="Performance Comparison", save_filename_prefix=f"../figures/exp_results_{tag}_auc")
