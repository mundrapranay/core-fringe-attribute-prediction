import numpy as np 
import random
from datetime import datetime
from scipy.sparse import csr_matrix
from numpy.linalg import norm

def expected_degree_imputation(adj_matrix, core_indices, fringe_indices, n1, n2, p_in, p_out):
    # Calculate expected degrees for fringe nodes
    expected_degrees = [((len(core_indices) + len(fringe_indices) - 1)/(n1+n2-1)) * (((n1 * (n1 -1) + n2*(n2-1))/(n1 +n2))*p_in + (2*n1*n2*p_out)/(n1+n2))] * len(fringe_indices)
    
    # Get actual degrees to core
    deg_to_core = np.array(adj_matrix[fringe_indices, :][:, core_indices].sum(axis=1)).flatten()
    
    # Calculate remaining degrees needed for each fringe node
    remaining_degrees = [int(max(0, exp - obs)) for exp, obs in zip(expected_degrees, deg_to_core)]
    
    # Sort fringe indices for deterministic processing
    fringe_indices = sorted(fringe_indices)
    
    # Convert to LIL format for efficient modification
    adj_lil = adj_matrix.tolil()
    seed = random.seed(datetime.now().timestamp())
    np.random.seed(seed)
    # For each fringe node
    for i, idx in enumerate(fringe_indices):
        # Get remaining fringe nodes (to avoid double counting)
        remaining_fringe = fringe_indices[i+1:]
        
        # Calculate how many edges we need to add
        edges_to_add = remaining_degrees[i]
        if edges_to_add > 0 and len(remaining_fringe) > 0:
            # Randomly select edges to add
            edges = np.random.choice(remaining_fringe, 
                                   size=min(edges_to_add, len(remaining_fringe)), 
                                   replace=False)
            
            # Add edges in both directions (undirected graph)
            for j in edges:
                adj_lil[idx, j] = 1
                adj_lil[j, idx] = 1
                
                # Update remaining degrees for the connected node
                j_idx = fringe_indices.index(j)
                remaining_degrees[j_idx] = max(0, remaining_degrees[j_idx] - 1)
    
    # Convert back to CSR format
    return adj_lil.tocsr() 

# def expected_degree_imputation(adj_matrix, core_indices, fringe_indices, n1, n2, p_in, p_out, metadata):
#     """
#     Improved version that uses class information from core nodes to better predict fringe-fringe edges.
#     We don't know fringe node classes, but we can use the core node classes to inform our predictions.
#     """
#     # Get class information for core nodes
#     core_classes = metadata[:, 1][core_indices]  # Assuming metadata contains class labels (0 or 1)
    
#     # Calculate expected degrees for fringe nodes based on their connections to core
#     expected_degrees = []
#     for fringe_idx in fringe_indices:
#         # Get this fringe node's connections to core nodes
#         fringe_to_core = adj_matrix[fringe_idx, core_indices].toarray().flatten()
        
#         # Count how many connections this fringe node has to each class of core nodes
#         connections_to_class1 = np.sum(fringe_to_core[core_classes == 0])
#         connections_to_class2 = np.sum(fringe_to_core[core_classes == 1])
        
#         # Calculate expected degree based on observed connections
#         # If a fringe node connects more to class1 core nodes, it's more likely to be class1
#         # and vice versa
#         if connections_to_class1 > connections_to_class2:
#             # More likely to be class1, so use class1 probabilities
#             expected_deg = (n1 * p_in + n2 * p_out)
#         else:
#             # More likely to be class2, so use class2 probabilities
#             expected_deg = (n1 * p_out + n2 * p_in)
            
#         expected_degrees.append(expected_deg)
    
#     # Get actual degrees to core
#     deg_to_core = np.array(adj_matrix[fringe_indices, :][:, core_indices].sum(axis=1)).flatten()
    
#     # Calculate remaining degrees needed for each fringe node
#     remaining_degrees = [int(max(0, exp - obs)) for exp, obs in zip(expected_degrees, deg_to_core)]
    
#     # Sort fringe indices for deterministic processing
#     fringe_indices = sorted(fringe_indices)
    
#     # Convert to LIL format for efficient modification
#     adj_lil = adj_matrix.tolil()
    
#     # For each fringe node
#     for i, idx in enumerate(fringe_indices):
#         # Get remaining fringe nodes (to avoid double counting)
#         remaining_fringe = fringe_indices[i+1:]
        
#         # Calculate how many edges we need to add
#         edges_to_add = remaining_degrees[i]
        
#         if edges_to_add > 0 and len(remaining_fringe) > 0:
#             # Calculate probabilities for each potential edge based on observed connections
#             edge_probs = []
#             for j in remaining_fringe:
#                 # Get both nodes' connections to core
#                 i_to_core = adj_matrix[idx, core_indices].toarray().flatten()
#                 j_to_core = adj_matrix[j, core_indices].toarray().flatten()
                
#                 # If both nodes connect similarly to core classes, they're more likely to be same class
#                 if np.corrcoef(i_to_core, j_to_core)[0,1] > 0:
#                     edge_probs.append(p_in)
#                 else:
#                     edge_probs.append(p_out)
            
#             # Normalize probabilities
#             edge_probs = np.array(edge_probs)
#             edge_probs = edge_probs / np.sum(edge_probs)
            
#             # Select edges with probability proportional to their similarity
#             edges = np.random.choice(remaining_fringe, 
#                                    size=min(edges_to_add, len(remaining_fringe)), 
#                                    replace=False,
#                                    p=edge_probs)
            
#             # Add edges in both directions (undirected graph)
#             for j in edges:
#                 adj_lil[idx, j] = 1
#                 adj_lil[j, idx] = 1
                
#                 # Update remaining degrees for the connected node
#                 j_idx = fringe_indices.index(j)
#                 remaining_degrees[j_idx] = max(0, remaining_degrees[j_idx] - 1)
    
#     # Convert back to CSR format
#     return adj_lil.tocsr() 



def compare_fringe_fringe_predictions(ff_true, ff_predicted):
    """
    Compare true and predicted fringe-fringe adjacency matrices.
    """
    # Convert to dense arrays for easier comparison
    true_dense = ff_true.toarray()
    pred_dense = ff_predicted.toarray()
    
    # Get upper triangle indices (excluding diagonal)
    n = len(true_dense)
    i, j = np.triu_indices_from(true_dense, k=1)  # k=1 to exclude diagonal
    
    # Get values from upper triangle
    true_upper = true_dense[i, j]
    pred_upper = pred_dense[i, j]
    
    # Calculate metrics
    tp = np.sum((true_upper == 1) & (pred_upper == 1))
    fp = np.sum((true_upper == 0) & (pred_upper == 1))
    fn = np.sum((true_upper == 1) & (pred_upper == 0))
    tn = np.sum((true_upper == 0) & (pred_upper == 0))
    
    total_possible_edges = len(true_upper)  # This will be correct now
    
    # Calculate metrics
    accuracy = (tp + tn) / total_possible_edges
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Edge density comparison
    true_density = np.sum(true_upper) / total_possible_edges
    pred_density = np.sum(pred_upper) / total_possible_edges
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'true_positive_rate': tp / (tp + fn) if (tp + fn) > 0 else 0,
        'false_positive_rate': fp / (fp + tn) if (fp + tn) > 0 else 0,
        'true_density': true_density,
        'predicted_density': pred_density,
        'total_edges_true': int(np.sum(true_upper)),
        'total_edges_predicted': int(np.sum(pred_upper))
    }



def fill_fringe_fringe_block_benson(adj_matrix,
                                    fringe_indices,
                                    method: str = 'cn',
                                    threshold: float = 0.5):
    """
    Fill the fringe–fringe block of adj_matrix by predicting edges
    (0/1) based on:
      - method='cn': common neighbors count
      - method='j':  jaccard similarity
    Any predicted score > threshold is set to 1, else 0.

    Parameters
    ----------
    adj_matrix : array-like or sparse matrix, shape (n, n)
        Original adjacency (0 for unknown/missing edges).
    fringe_indices : list[int]
        Indices of fringe nodes.
    method : {'cn', 'j'}, default='cn'
        'cn' for common neighbors, 'j' for Jaccard.
    threshold : float, default=0.0
        Cutoff: score > threshold → 1, else 0.

    Returns
    -------
    csr_matrix
        A new adjacency where the [fringe, fringe] block is 0/1 predictions
        and the rest is unchanged.
    """
    # Convert to dense numpy array for computations
    A = adj_matrix.toarray() if hasattr(adj_matrix, 'toarray') else np.array(adj_matrix)
    n = A.shape[0]

    # Binary neighbor matrix from ORIGINAL A
    B = (A > 0).astype(int)
    deg = B.sum(axis=1)  # degrees for Jaccard

    # Prepare output; start from original adjacency (float so we can assign)
    A_pred = A.astype(float).copy()

    # Loop over each unordered fringe pair once
    f = fringe_indices
    m = len(f)
    for idx in range(m):
        u = f[idx]
        nu = B[u]
        for jdx in range(idx+1, m):
            v = f[jdx]

            # skip if already an observed edge
            if B[u, v]:
                continue

            nv = B[v]
            inter = int((nu & nv).sum())

            if method == 'cn':
                score = inter
            elif method == 'jk':
                union = deg[u] + deg[v] - inter
                score = inter/union if union > 0 else 0.0
            else:
                raise ValueError(f"Unknown method '{method}'")

            # threshold to binary prediction
            pred = 1 if score > threshold else 0

            # fill symmetric entries
            A_pred[u, v] = pred
            A_pred[v, u] = pred

    return csr_matrix(A_pred)



def fill_fringe_fringe_block_class_cosine(adj_matrix,
                                            fringe_indices,
                                            core_indices,
                                            metadata,
                                            threshold: float = 0.5):
    """
    Fill the fringe–fringe block of adj_matrix by predicting edges (0/1) based on cosine similarity
    of core-class neighbor vectors. For each fringe node, create a 2D vector:
      [#core neighbors of class 1, #core neighbors of class 2]
    and add an edge if the cosine similarity between two fringe nodes' vectors is above threshold.

    Parameters
    ----------
    adj_matrix : array-like or sparse matrix, shape (n, n)
        Original adjacency (0 for unknown/missing edges).
    fringe_indices : list[int]
        Indices of fringe nodes.
    core_indices : list[int]
        Indices of core nodes.
    metadata : np.ndarray
        Node metadata, where column 1 is gender/class (1 or 2).
    threshold : float, default=0.5
        Cosine similarity cutoff: sim > threshold → 1, else 0.

    Returns
    -------
    csr_matrix
        A new adjacency where the [fringe, fringe] block is 0/1 predictions
        and the rest is unchanged.
    """

    A = adj_matrix.toarray() if hasattr(adj_matrix, 'toarray') else np.array(adj_matrix)
    n = A.shape[0]
    # Prepare output; start from original adjacency (float so we can assign)
    A_pred = A.astype(float).copy()
    # Get core node classes (gender)
    core_classes = metadata[core_indices].astype(int)
    class_labels = np.unique(core_classes)
    # For each fringe node, build 2D vector: [#core neighbors of class 1, #core neighbors of class 2]
    fringe_vecs = {}
    for u in fringe_indices:
        vec = np.zeros(2, dtype=int)
        neighbors = np.where(A[u, core_indices] > 0)[0]
        for idx in neighbors:
            cls = core_classes[idx]
            if cls == class_labels[0]:
                vec[0] += 1
            elif cls == class_labels[1]:
                vec[1] += 1
        fringe_vecs[u] = vec
    # Loop over each unordered fringe pair once
    f = fringe_indices
    m = len(f)
    for idx in range(m):
        u = f[idx]
        vec_u = fringe_vecs[u]
        for jdx in range(idx+1, m):
            v = f[jdx]
            # skip if already an observed edge
            if A_pred[u, v] > 0:
                continue
            vec_v = fringe_vecs[v]
            # Compute cosine similarity
            if norm(vec_u) == 0 or norm(vec_v) == 0:
                sim = 0.0
            else:
                sim = np.dot(vec_u, vec_v) / (norm(vec_u) * norm(vec_v))
            # threshold to binary prediction
            pred = 1 if sim > threshold else 0
            # fill symmetric entries
            A_pred[u, v] = pred
            A_pred[v, u] = pred
    return csr_matrix(A_pred)

def logistic_regression_link_prediction(adj_matrix, core_indices, fringe_indices, method='eigenValue2', threshold=0.5, lr_kwargs=None):
    """
    Predict fringe-fringe edges using LR trained on core-core and core-fringe links.
    Features are determined by the 'method' parameter.
    """
    import numpy as np
    from scipy.sparse import csr_matrix
    from sklearn.linear_model import LogisticRegression
    from numpy.linalg import eigvalsh

    if lr_kwargs is None:
        lr_kwargs = {'solver': 'liblinear', 'max_iter': 1000}

    A = adj_matrix.toarray() if hasattr(adj_matrix, 'toarray') else np.array(adj_matrix)
    n = A.shape[0]
    A_pred = A.copy()

    # 1. Build training set (core-core and core-fringe pairs)
    X_train = []
    y_train = []
    # Core-core
    for i, u in enumerate(core_indices):
        for j, v in enumerate(core_indices):
            if u >= v:
                continue
            feature = extract_link_feature(A, u, v, method)
            # print(feature)
            X_train.append([feature])
            y_train.append(A[u, v])
    # Core-fringe
    for u in core_indices:
        for v in fringe_indices:
            feature = extract_link_feature(A, u, v, method)
            # print(feature)
            X_train.append([feature])
            y_train.append(A[u, v])

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    # 2. Train LR
    model = LogisticRegression(**lr_kwargs)
    model.fit(X_train, y_train)

    # 3. Predict fringe-fringe edges
    for i, u in enumerate(fringe_indices):
        for j, v in enumerate(fringe_indices):
            if u >= v:
                continue
            feature = extract_link_feature(A, u, v, method)
            prob = model.predict_proba([[feature]])[0, 1]
            # print(prob)
            if prob > threshold:
                A_pred[u, v] = 1
                A_pred[v, u] = 1
            else:
                A_pred[u, v] = 0
                A_pred[v, u] = 0

    return csr_matrix(A_pred)

def extract_link_feature(A, u, v, method):
    """
    Extracts a feature for the link (u, v) based on the specified method.
    Supported methods:
      - 'eigenValue2': 2nd smallest eigenvalue (Fiedler value)
      - 'eigenValue3': 3rd smallest eigenvalue
      - 'nonZeroEigen': smallest positive eigenvalue
    """
    import numpy as np
    from numpy.linalg import eigvalsh

    # subA = A[np.ix_([u, v], [u, v])]
    # print(subA)
    # print(subA.shape)
    # Create submatrix with only intersection of neighbors of u and v
    subA = np.zeros(A.shape)
    
    # # Get neighbors of u and v
    # neighbors_u = np.where(A[u, :] > 0)[0]
    # neighbors_v = np.where(A[v, :] > 0)[0]
    
    # # Find intersection of neighbors
    # intersection = np.intersect1d(neighbors_u, neighbors_v)
    
    # # Set intersection points to 1
    # for node in intersection:
    #     subA[u, node] = 1
    #     subA[node, u] = 1
    #     subA[v, node] = 1
    #     subA[node, v] = 1
    subA[u, :] = A[u, :]  # Entire row u
    subA[v, :] = A[v, :]  # Entire row v
    subA[:, u] = A[:, u]  # Entire column u
    subA[:, v] = A[:, v]  # Entire column v
    # Laplacian
    L = np.diag(subA.sum(axis=1)) - subA
    eigvals = np.sort(eigvalsh(L))

    if method == 'eigenValue2':
        # Return the 2nd smallest eigenvalue (Fiedler value)
        return eigvals[1] if len(eigvals) > 1 else 0.0
    elif method == 'eigenValue3':
        # Return the 3rd smallest eigenvalue
        return eigvals[2] if len(eigvals) > 2 else 0.0
    elif method == 'nonZeroEigen':
        # Return the smallest positive eigenvalue
        pos_eigvals = eigvals[eigvals > 1e-8]
        return pos_eigvals[0] if len(pos_eigvals) > 0 else 0.0
    else:
        raise NotImplementedError(f"Feature method '{method}' not implemented.")

