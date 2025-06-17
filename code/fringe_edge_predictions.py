import numpy as np 
import random
from datetime import datetime

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
