import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
import random
import networkx as nx
from gensim.models import Word2Vec
from scipy.sparse import csr_matrix
from collections import Counter
from sklearn.utils import resample
from fringe_edge_predictions import *

def estimate_expected_degree_from_core(adj_matrix, core_indices, fringe_indices):
    """
    For each fringe node, compute its expected fringe–fringe degree
    as the *average* degree of the core nodes it connects to.
    If a fringe node has no core neighbors, fall back to the global
    average core degree.
    """
    core_deg = np.array(adj_matrix[core_indices, :].sum(axis=1)).flatten()
    global_avg = core_deg.mean()

    expected = np.zeros(len(fringe_indices), dtype=float)

    for i, f in enumerate(fringe_indices):
        connections = adj_matrix[core_indices, f].toarray().flatten()
        neighbor_positions = np.nonzero(connections)[0]
        if neighbor_positions.size > 0:
            expected[i] = core_deg[neighbor_positions].mean()
        else:
            expected[i] = global_avg

    return expected


def estimate_expected_degree_iid_core(adj_matrix, core_indices, fringe_indices):
    """
    For each fringe node, estimate its total degree in the full graph
    using its observed degree to the IID-sampled core.

    Parameters:
    - adj_matrix: adjacency matrix (scipy.sparse or np.ndarray)
    - core_indices: indices of core nodes (IID sample)
    - fringe_indices: indices of fringe nodes

    Returns:
    - expected: np.array of shape (len(fringe_indices),), estimated total degree for each fringe node
    """
    n_core = len(core_indices)
    n_total = adj_matrix.shape[0]

    # Degree to core for each fringe node
    deg_to_core = np.array(adj_matrix[fringe_indices, :][:, core_indices].sum(axis=1)).flatten()
    # Scale up to estimate total degree
    expected = deg_to_core * (n_total / n_core)
    return expected


def iid_sbm_expected_degree(adj_matrix, core_indices, fringe_indices, n1, n2, p_in, p_out):
    expected_degrees = [((len(core_indices) + len(fringe_indices) - 1)/(n1+n2-1)) * (((n1 * (n1 -1) + n2*(n2-1))/(n1 +n2))*p_in + (2*n1*n2*p_out)/(n1+n2))] * len(fringe_indices)
    deg_to_core = np.array(adj_matrix[fringe_indices, :][:, core_indices].sum(axis=1)).flatten()
    return [(e-d) for e,d in zip(expected_degrees, deg_to_core)]


def naive_estimated_degree(adj_matrix, core_indices, fringe_indices):
    return [np.array(adj_matrix[core_indices, :].sum(axis=1)).flatten().mean()] * len(fringe_indices)

def link_logistic_regression_pipeline(adj_matrix, core_indices, fringe_indices, metadata, core_only=False, lr_kwargs=None, expected_degree=False, iid_core=False, naive_degree=False, sbm=False, ff=False, ff_value=None, p_core_fringe=0.0, p_fringe_fringe=0.0, return_auc_ci=False, plot_auc_ci=False):
    # Get gender and dorm information
    gender = metadata[:, 1].astype(int)  # Convert to integer
    dorm = metadata[:, 4]

    # Create core-only adjacency matrix
    if ff:
        adj_matrix[fringe_indices, :][:, fringe_indices] = ff_value

    if core_only:
        X_train = adj_matrix[core_indices, :][:, core_indices]
        y_train = gender[core_indices]
        X_test = adj_matrix[fringe_indices, :][:, core_indices]
        print("\n Feature Space (Core-Core only)")
        print(f"X_train shape: {X_train.shape}")
        print(f"y_train shape: {y_train.shape}")
    else:
        X_train = adj_matrix[core_indices, :]
        y_train = gender[core_indices]
        X_test = adj_matrix[fringe_indices, :]
        print("\n Feature Space (Core-Fringe)")
        print(f"X_train shape: {X_train.shape}")
        print(f"y_train shape: {y_train.shape}")
    
    y_test = gender[fringe_indices]
    seed = 123
    unique_train_classes = np.unique(y_train)
    print(f"Unique training classes: {unique_train_classes}")
    if unique_train_classes.size < 2:
        print("Not enough unique classes for training. Skipping logistic regression.")
        return
    
    # # Analyze feature space
    # print("\nFeature Space Analysis:")
    # # For each class, calculate mean degree to all nodes
    # for class_label in unique_train_classes:
    #     class_mask = (y_train == class_label)
    #     mean_degree = X_train[class_mask].sum(axis=1).mean()
    #     print(f"Mean degree of class {class_label} nodes: {mean_degree:.2f}")
    
    # # Calculate homophily (degree within same class)
    # for class_label in unique_train_classes:
    #     class_mask = (y_train == class_label)
    #     if core_only:
    #         # In core-only, both axes are core nodes, so use class_mask for both
    #         class_submatrix = X_train[class_mask][:, class_mask]
    #         mean_same_class_degree = class_submatrix.sum(axis=1).mean()
    #     else:
    #         # In core-fringe, columns are all nodes, so use core_indices for columns
    #         class_submatrix = X_train[class_mask][:, core_indices]
    #         same_class_mask = (y_train == class_label)
    #         mean_same_class_degree = class_submatrix[:, same_class_mask].sum(axis=1).mean()
    #     print(f"Mean degree of class {class_label} nodes to same class: {mean_same_class_degree:.2f}")
    
    # Train logistic regression model
    model = LogisticRegression(**lr_kwargs, random_state=seed)
    model.fit(X_train, y_train)
    beta = model.coef_.flatten()
    print(f"\nModel Analysis:")
    print(f"Number of non-zero coefficients: {np.count_nonzero(beta)}")
    print(f"Mean absolute coefficient: {np.mean(np.abs(beta)):.4f}")
    print(f"Max coefficient: {np.max(np.abs(beta)):.4f}")
    print(f"Min coefficient: {np.min(np.abs(beta)):.4f}")
    print(f"Max coefficient (No-Abs): {np.max(beta):.4f}")
    print(f"Min coefficient (No-Abs): {np.min(beta):.4f}")
    
    # Make predictions on test set
    if expected_degree:
        if naive_degree:
            expected_degree = naive_estimated_degree(adj_matrix, core_indices, fringe_indices)
        elif iid_core:
            expected_degree = estimate_expected_degree_iid_core(adj_matrix, core_indices, fringe_indices)
        elif sbm:
            # expected_degree = estimate_expected_degree_sbm(core_indices, fringe_indices, p_core_fringe, p_fringe_fringe=p_fringe_fringe)
            # expected_degree = iid_sbm_expected_degree(adj_matrix, core_indices, fringe_indices, 500, 500, 0.15, 0.1)
            adj_matrix_imputed = expected_degree_imputation(adj_matrix, core_indices, fringe_indices, 500, 500, 0.15, 0.1)
            X_test = adj_matrix_imputed[fringe_indices, :]
        else:
            expected_degree = estimate_expected_degree_from_core(adj_matrix, core_indices, fringe_indices)
        # for i, deg in enumerate(expected_degree):
        #     k = int(round(deg))
        #     if k > 0:
        #         chosen_cols = np.random.choice(fringe_indices, size=min(k, len(fringe_indices)), replace=False)
        #         X_test[i, chosen_cols] = 1
    
    y_test_pred = model.predict(X_test)
    y_test_scores = model.predict_proba(X_test)
    
    # Verify class order and AUC calculation
    print("\nClass Order Verification:")
    print(f"Model classes_: {model.classes_}")  # Order of classes in the model
    print(f"Unique test classes: {np.unique(y_test)}")  # Classes in test set
    print(f"Class distribution in test: {dict(Counter(y_test))}")
    print(f"Prediction distribution: {dict(Counter(y_test_pred))}")
    
    # Calculate AUC for each class
    for i, class_label in enumerate(model.classes_):
        class_auc = roc_auc_score(y_test == class_label, y_test_scores[:, i])
        print(f"AUC for class {class_label}: {class_auc:.4f}")
    
    # Use the correct class index for AUC
    positive_class_idx = np.where(model.classes_ == 2)[0][0]  # Assuming 2 is our positive class
    auc = roc_auc_score(y_test, y_test_scores[:, positive_class_idx])
    accuracy = accuracy_score(y_test, y_test_pred)
    
    # Compute AUC confidence interval
    auc_lower, auc_upper = auc_confidence_interval(y_test, y_test_scores[:, positive_class_idx])
    print(f"AUC 95% CI: [{auc_lower:.3f}, {auc_upper:.3f}]")
    
    print(f"\nFinal Results:")
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test ROC AUC: {auc:.4f}")
    print(f"Training class distribution: {dict(Counter(y_train))}")
    print(f"Test class distribution: {dict(Counter(y_test))}")
    print(f"X_train sparsity: {1 - X_train.nnz / (X_train.shape[0] * X_train.shape[1]):.4f}")
    print(f"X_test sparsity: {1 - X_test.nnz / (X_test.shape[0] * X_test.shape[1]):.4f}")
    
    if return_auc_ci:
        return beta, accuracy, auc, (auc_lower, auc_upper)
    else:
        return beta, accuracy, auc

def random_guesser(fringe_indices, metadata, seed):
    """
    Simulate a random guesser for the fringe nodes' gender.
    For each trial, randomly guess gender (1 or 2) for each fringe node.
    Compute AUC for each trial and return the mean and 95% CI using auc_confidence_interval.
    """
    rng = np.random.RandomState(seed)
    gender = metadata[:, 1].astype(int)
    y_true = gender[fringe_indices]
    y_scores = rng.rand(len(fringe_indices))
    auc = roc_auc_score(y_true, y_scores)
    lower, upper = auc_confidence_interval(y_true, y_scores, n_bootstraps=1000, random_seed=seed)
    print(f"Random Guesser: Mean AUC = {auc:.4f}, 95% CI = [{lower:.4f}, {upper:.4f}]")
    return auc, (lower, upper)

def alias_setup(probs):
    """
    Construct alias tables (J, q) for non-uniform sampling in O(n).
    """
    K = len(probs)
    q = np.zeros(K)
    J = np.zeros(K, dtype=np.int32)
    smaller, larger = [], []
    prob = np.array(probs) * K

    for idx, p in enumerate(prob):
        if p < 1.0:
            smaller.append(idx)
        else:
            larger.append(idx)

    while smaller and larger:
        small = smaller.pop()
        large = larger.pop()
        q[small] = prob[small]
        J[small] = large
        prob[large] = prob[large] - (1.0 - prob[small])
        if prob[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)

    for leftover in larger + smaller:
        q[leftover] = 1.0
    return J, q

def alias_draw(J, q):
    """
    Draw sample from a non-uniform discrete distribution using alias tables.
    """
    K = len(J)
    i = int(np.floor(np.random.rand() * K))
    return i if np.random.rand() < q[i] else J[i]

class PureNode2Vec:
    def __init__(self, graph, p=1.0, q=1.0):
        """
        graph: a NetworkX graph (undirected, unweighted)
        p, q: Node2Vec return and in-out parameters
        """
        self.G = graph
        self.p = p
        self.q = q
        self.alias_nodes = {}
        self.alias_edges = {}
        self._preprocess()

    def _get_alias_node(self, node):
        """
        Precompute alias sampling for a node's neighbors (uniform).
        """
        neighbors = list(self.G.neighbors(node))
        if not neighbors:
            return None  # no neighbors—walks will just stop
        probs = [1.0] * len(neighbors)
        # normalize (uniform)
        norm = float(sum(probs))
        probs = [p / norm for p in probs]
        return alias_setup(probs)

    def _get_alias_edge(self, src, dst):
        """
        Precompute alias sampling for transitions from src->dst.
        """
        neighbors = list(self.G.neighbors(dst))
        if not neighbors:
            return None
        probs = []
        for nbr in neighbors:
            if nbr == src:
                w = 1.0 / self.p
            elif self.G.has_edge(nbr, src):
                w = 1.0
            else:
                w = 1.0 / self.q
            probs.append(w)
        norm = float(sum(probs))
        if norm <= 0.0:
            # fallback to uniform if something went wrong
            probs = [1.0 / len(neighbors)] * len(neighbors)
        else:
            probs = [w / norm for w in probs]
        return alias_setup(probs)

    def _preprocess(self):
        """
        Precompute all alias tables for nodes and edges.
        """
        # Nodes
        for node in self.G.nodes():
            self.alias_nodes[node] = self._get_alias_node(node)
        # Edges
        for src, dst in self.G.edges():
            self.alias_edges[(src, dst)] = self._get_alias_edge(src, dst)
            self.alias_edges[(dst, src)] = self._get_alias_edge(dst, src)

    def simulate_walk(self, walk_length, start_node):
        """
        Generate a single walk of length walk_length from start_node.
        """
        walk = [start_node]
        while len(walk) < walk_length:
            curr = walk[-1]
            nbrs = list(self.G.neighbors(curr))
            if not nbrs:
                break
            if len(walk) == 1:
                node_alias = self.alias_nodes.get(curr)
                if node_alias is None:
                    break
                J, q = node_alias
                idx = alias_draw(J, q)
                walk.append(nbrs[idx])
            else:
                prev = walk[-2]
                edge_alias = self.alias_edges.get((prev, curr))
                if edge_alias is None:
                    break
                J, q = edge_alias
                idx = alias_draw(J, q)
                walk.append(nbrs[idx])
        return walk

    def generate_walks(self, num_walks, walk_length):
        """
        Generate num_walks walks per node.
        """
        walks = []
        nodes = list(self.G.nodes())
        for _ in range(num_walks):
            random.shuffle(nodes)
            for n in nodes:
                walks.append(self.simulate_walk(walk_length, n))
        return walks

def node2vec_embeddings_gensim(adj_matrix,
                               dimensions=64,
                               walk_length=30,
                               num_walks=10,
                               window_size=5,
                               p=1.0,
                               q=1.0,
                               workers=4,
                               alpha=0.025,
                               seed=42):
    """
    1. Build NetworkX graph from adjacency.
    2. Run our PureNode2Vec to get walks.
    3. Train Word2Vec on walks.
    4. Return embeddings dict {node: vector}.
    """
    # 1. Build graph
    G_nx = nx.from_scipy_sparse_array(csr_matrix(adj_matrix))
    n2v = PureNode2Vec(G_nx, p=p, q=q)

    # 2. Generate walks
    walks = n2v.generate_walks(num_walks=num_walks, walk_length=walk_length)
    str_walks = [[str(n) for n in walk] for walk in walks]

    # 3. Word2Vec
    model = Word2Vec(
        sentences=str_walks,
        vector_size=dimensions,
        window=window_size,
        min_count=0,
        sg=1,
        workers=workers,
        alpha=alpha,
        seed=seed
    )
    # 4. Extract embeddings
    emb = {int(node): model.wv[node] for node in model.wv.key_to_index}
    return emb


def node2vec_logistic_regression_pipeline(adj_matrix, core_indices, fringe_indices, metadata, core_only=False, lr_kwargs=None, embed_kwargs=None, seed=42):
    if embed_kwargs is None:
        embed_kwargs = {}
    if lr_kwargs is None:
        lr_kwargs = {'solver':'liblinear', 'max_iter':1000}

    # Generate embeddings
    embeddings = node2vec_embeddings_gensim(adj_matrix, seed=seed, **embed_kwargs)
    gender = metadata[:, 1]
    # Build feature matrices 
    X_core   = np.vstack([embeddings[i] for i in core_indices])
    y_core   = gender[core_indices]
    X_fringe = np.vstack([embeddings[i] for i in fringe_indices])
    y_fringe = gender[fringe_indices]

    # Train & evaluate logistic regression 
    model = LogisticRegression(**lr_kwargs, random_state=seed)
    model.fit(X_core, y_core)
    beta = model.coef_.flatten()
    print(f"Number of non-zero coefficients: {np.count_nonzero(beta)}")
    print(f"Mean absolute coefficient: {np.mean(np.abs(beta)):.4f}")
    print(f"Max coefficient: {np.max(np.abs(beta)):.4f}")
    print(f"Min coefficient: {np.min(np.abs(beta)):.4f}")
    print(f"Max coefficient (No-Abs): {np.max(beta):.4f}")
    print(f"Min coefficient (No-Abs): {np.min(beta):.4f}")
    
    y_test_pred = model.predict(X_fringe)
    y_test_scores = model.predict_proba(X_fringe)
    accuracy = accuracy_score(y_fringe, y_test_pred)
    auc = roc_auc_score(y_fringe, y_test_scores[:, 1])
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test ROC AUC: {auc:.4f}")
    print(f"Training class distribution: {np.bincount(y_core)}")
    print(f"Test class distribution: {np.bincount(y_fringe)}")
    print(f"X_train sparsity: {1 - X_core.nnz / (X_core.shape[0] * X_core.shape[1]):.4f}")
    print(f"X_test sparsity: {1 - X_fringe.nnz / (X_fringe.shape[0] * X_fringe.shape[1]):.4f}")
    print(f"Model classes: {model.classes_}")
    print(f"Prediction distribution: {np.bincount(y_test_pred)}")
    return beta, accuracy, auc

def estimate_expected_degree_sbm(core_indices, fringe_indices, p_core_fringe, p_fringe_fringe=0.0):
    """
    For each fringe node in an SBM, estimate its expected total degree
    based on SBM parameters.

    Parameters:
    - core_indices: indices of core nodes
    - fringe_indices: indices of fringe nodes
    - p_core_fringe: probability of edge between core and fringe
    - p_fringe_fringe: probability of edge between fringe nodes (default 0.0)

    Returns:
    - expected: np.array of shape (len(fringe_indices),), expected total degree for each fringe node
    """
    n_core = len(core_indices)
    n_fringe = len(fringe_indices)
    expected = np.full(len(fringe_indices), n_core * p_core_fringe + n_fringe * p_fringe_fringe)
    return expected

def auc_confidence_interval(y_true, y_scores, n_bootstraps=1000, random_seed=42):
    rng = np.random.RandomState(random_seed)
    bootstrapped_scores = []
    for i in range(n_bootstraps):
        indices = rng.randint(0, len(y_true), len(y_true))
        if len(np.unique(y_true[indices])) < 2:
            # skip if only one class in the sample
            continue
        score = roc_auc_score(y_true[indices], y_scores[indices])
        bootstrapped_scores.append(score)
    sorted_scores = np.sort(bootstrapped_scores)
    lower = sorted_scores[int(0.025 * len(sorted_scores))]
    upper = sorted_scores[int(0.975 * len(sorted_scores))]
    return lower, upper
