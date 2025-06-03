import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
import random
import networkx as nx
from gensim.models import Word2Vec
from scipy.sparse import csr_matrix


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


def naive_estimated_degree(adj_matrix, core_indices, fringe_indices):
    return [np.array(adj_matrix[core_indices, :].sum(axis=1)).flatten().mean()] * len(fringe_indices)

def link_logistic_regression_pipeline(adj_matrix, core_indices, fringe_indices, metadata, core_only=False, lr_kwargs=None, expected_degree=False, iid_core=False, naive_degree=False, sbm=False, p_core_fringe=0.0, p_fringe_fringe=0.0):
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
    seed = 123
    unique_train_classes = np.unique(y_train)
    print(f"Unique training classes: {unique_train_classes}")
    if unique_train_classes.size < 2:
        print("Not enough unique classes for training. Skipping logistic regression.")
        return
    
    # Train logistic regression model
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
    if expected_degree:
        if naive_degree:
            expected_degree = naive_estimated_degree(adj_matrix, core_indices, fringe_indices)
        elif iid_core:
            expected_degree = estimate_expected_degree_iid_core(adj_matrix, core_indices, fringe_indices)
        elif sbm:
            expected_degree = estimate_expected_degree_sbm(core_indices, fringe_indices, p_core_fringe, p_fringe_fringe=p_fringe_fringe)
        else:
            expected_degree = estimate_expected_degree_from_core(adj_matrix, core_indices, fringe_indices)
        # first_fringe_col = fringe_indices[0]
        # X_test[np.arange(len(fringe_indices)), first_fringe_col] = expected_degree
        for i, deg in enumerate(expected_degree):
            # For each fringe node (row i)
            k = int(round(deg))
            if k > 0:
                # Randomly select k columns from the fringe_indices (fringe-fringe block)
                chosen_cols = np.random.choice(fringe_indices, size=min(k, len(fringe_indices)), replace=False)
                X_test[i, chosen_cols] = 1
    
    y_test_pred = model.predict(X_test)
    y_test_scores = model.predict_proba(X_test)
    accuracy = accuracy_score(y_test, y_test_pred)
    auc = roc_auc_score(y_test, y_test_scores[:, 1])
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test ROC AUC: {auc:.4f}")
    return beta, accuracy, auc



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
