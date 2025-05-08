import os
import random
from os import listdir
from os.path import join as path_join
import itertools
import numpy as np
# from define_paths import * 
from data_loader import *
import random
# 1) Patch scipy.linalg
import scipy.linalg as la
import numpy as np
if not hasattr(la, 'triu'):
    la.triu = lambda m, k=0: np.triu(m, k)
import networkx as nx
from gensim.models import Word2Vec
from sklearn.metrics import roc_auc_score
import math

fb_code_path = '/Users/pranaymundra/Desktop/research_code/core-fringe-attribute-prediction/data/fb100/' ## directory path of raw FB100 .mat files 


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

def predict_attribute_with_node2vec(adj_matrix,
                                    core_indices,
                                    fringe_indices,
                                    labels,
                                    embed_kwargs=None,
                                    lr_kwargs=None,
                                    seed=42):
    """
    1. Generate Node2Vec embeddings via Gensim.
    2. Build feature matrices for core and fringe nodes.
    3. Train LogisticRegression on core embeddings.
    4. Predict and return accuracy on fringe.
    """
    if embed_kwargs is None:
        embed_kwargs = {}
    if lr_kwargs is None:
        lr_kwargs = {'solver':'liblinear', 'max_iter':1000}

    # Generate embeddings
    embeddings = node2vec_embeddings_gensim(adj_matrix, seed=seed, **embed_kwargs)

    # Build feature matrices 
    X_core   = np.vstack([embeddings[i] for i in core_indices])
    y_core   = labels[core_indices]
    X_fringe = np.vstack([embeddings[i] for i in fringe_indices])
    y_fringe = labels[fringe_indices]

    # Train & evaluate logistic regression 
    clf = LogisticRegression(**lr_kwargs, random_state=seed)
    clf.fit(X_core, y_core)
    y_pred = clf.predict(X_fringe)
    acc = accuracy_score(y_fringe, y_pred)  # :contentReference[oaicite:0]{index=0}

    print(f"Fringe accuracy: {acc:.2%}")
    return acc, clf

def predict_attribute_with_node2vec_auc_classwise(
    adj_matrix,
    core_indices,
    fringe_indices,
    labels,
    percentages,
    embed_kwargs=None,
    lr_kwargs=None,
    seed=42
):
    """
    For each fraction p in 'percentages':
      1. Generate Node2Vec embeddings once.
      2. Sample p·|core| of core nodes.
      3. Train LR on those core embeddings.
      4. Compute one-vs-rest AUC on the fringe for each class.
    
    Returns:
      auc_dict: dict mapping p → tuple of class-wise AUCs (or None if skipped).
    """
    if embed_kwargs is None:
        embed_kwargs = {}
    if lr_kwargs is None:
        lr_kwargs = {'solver':'liblinear', 'max_iter':1000}

    # 1) Generate embeddings once
    embeddings = node2vec_embeddings_gensim(adj_matrix, seed=seed, **embed_kwargs)

    
    # Precompute fringe features & labels
    X_fringe = np.vstack([embeddings[n] for n in fringe_indices])
    y_fringe = labels[fringe_indices]

    np.random.seed(seed)
    random.seed(seed)

    auc_dict = {}
    n_core = len(core_indices)

    for p in percentages:
        k = math.ceil(p * n_core)
        sampled_core = np.random.choice(core_indices, size=k, replace=False)

        # Build training set
        X_train = np.vstack([embeddings[n] for n in sampled_core])
        y_train = labels[sampled_core]

        # Skip if only one class in training
        if len(np.unique(y_train)) < 2:
            print(f"Skipping p={p:.2f}: only one class in sampled core")
            auc_dict[p] = None
            continue

        # Train logistic regression
        clf = LogisticRegression(**lr_kwargs, random_state=seed)
        clf.fit(X_train, y_train)

        # Predict probabilities for each class on the fringe
        y_scores = clf.predict_proba(X_fringe)
        classes = clf.classes_

        # Compute one-vs-rest AUC for each class
        aucs = []
        print(f"Percentage of labelled core nodes: {p*100:.0f}%")
        for idx, c in enumerate(classes):
            # binary ground truth for class c
            y_true_bin = (y_fringe == c).astype(int)
            # if fringe has no samples of this class, skip
            if np.unique(y_true_bin).size < 2:
                print(f"  Class {c} AUC: N/A (insufficient samples)")
                aucs.append(None)
            else:
                auc_score = roc_auc_score(y_true_bin, y_scores[:, idx])
                print(f"  Class {c} AUC: {auc_score*100:.2f}%")
                aucs.append(auc_score)

        # Store tuple of per-class AUCs
        auc_dict[p] = tuple(aucs)

    return auc_dict

def hyperparameter_search_node2vec_gensim(
    adj_matrix,
    core_indices,
    fringe_indices,
    labels,
    param_grid,
    lr_C_list=[1.0],
    seed=42
):
    """
    Grid-search over Gensim-based Node2Vec parameters (and optionally LR C).
    
    Parameters:
      - adj_matrix: scipy.sparse adjacency matrix
      - core_indices, fringe_indices: arrays of node IDs for train/test
      - labels: 1D array of node labels (binary or multiclass)
      - param_grid: dict mapping embedding-param names to lists of values, e.g.:
          {
            'dimensions': [32,64],
            'walk_length': [40,80],
            'num_walks': [5,10],
            'window_size': [5,10],
            'p': [0.5,1.0],
            'q': [1.0,2.0],
            'alpha': [0.01,0.025],
          }
      - lr_C_list: list of inverse‑regularization strengths to try for LogisticRegression
      - seed: random seed for reproducibility
    
    Returns:
      - best_combo: dict of best embedding params + best C
      - best_acc: highest fringe accuracy
      - all_results: list of tuples (combo_dict, accuracy)
    """
    keys = list(param_grid.keys())
    values = [param_grid[k] for k in keys]
    
    best_acc = -1.0
    best_combo = None
    all_results = []
    
    for combo in itertools.product(*values):
        embed_kwargs = dict(zip(keys, combo))
        
        for C in lr_C_list:
            lr_kwargs = {'C': C, 'solver': 'liblinear', 'max_iter': 500}
            
            print(f"\nEvaluating embed_params={embed_kwargs}, C={C}")
            try:
                acc, _ = predict_attribute_with_node2vec(
                    adj_matrix=adj_matrix,
                    core_indices=core_indices,
                    fringe_indices=fringe_indices,
                    labels=labels,
                    embed_kwargs=embed_kwargs,
                    lr_kwargs=lr_kwargs,
                    seed=seed
                )
            except Exception as e:
                print(" → Error: ", e)
                acc = 0.0
            
            combo_dict = embed_kwargs.copy()
            combo_dict['C'] = C
            all_results.append((combo_dict, acc))
            
            if acc > best_acc:
                best_acc = acc
                best_combo = combo_dict.copy()
    
    print("\n=== Hyperparameter Search Complete ===")
    print("Best params:", best_combo)
    print(f"Best fringe accuracy: {best_acc:.2%}")
    return best_combo, best_acc, all_results

def perform_experiment_for_multi_dorm_core_node2vec(adj_matrix, metadata, dorm_ids, percentages, seed=None):
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
    
    lr_kwargs = {'C': 0.1, 'solver': 'liblinear', 'max_iter': 1000}
    embed_kwargs={'dimensions': 32, 'walk_length': 40, 'num_walks': 5, 'window_size': 5, 'p': 0.5, 'q': 1.0, 'alpha': 0.025}
    # results_multi = predict_attribute_with_node2vec_auc_classwise(
    #     adj_matrix=multi_dorm_graph,
    #     core_indices=multi_dorm_core_indices,
    #     fringe_indices=multi_dorm_fringe_indices,
    #     labels=metadata[:,1],
    #     percentages=percentages,
    #     embed_kwargs=embed_kwargs,
    #     lr_kwargs=lr_kwargs,
    #     seed=42
    # )
    results_multi, betas_multi = link_lr_with_expected_fringe_degree_auc(
        adj_matrix=multi_dorm_graph,
        core_indices=multi_dorm_core_indices, 
        fringe_indices=multi_dorm_fringe_indices, 
        y_core=multi_gender_core, 
        y_fringe=multi_gender_fringe,
        percentages=percentages,
        lr_kwargs=lr_kwargs,
        seed=42
    )

    results_multi_core, betas_multi_core = link_logistic_regression_core_only_auc(
        adj_matrix=multi_dorm_graph,
        core_indices=multi_dorm_core_indices, 
        fringe_indices=multi_dorm_fringe_indices, 
        y_core=multi_gender_core, 
        y_fringe=multi_gender_fringe,
        percentages=percentages,
        lr_kwargs=lr_kwargs,
        seed=42
    )

    # plot_beta_vectors(betas_multi, "Yale_MultiDorm")
    plot_beta_comparison(betas_multi, betas_multi_core, "Yale_MultiDorm")
    
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
    
    # results_normal = predict_attribute_with_node2vec_auc_classwise(
    #     adj_matrix=adj_matrix, 
    #     core_indices=random_core_indices,
    #     fringe_indices=random_fringe_indices,
    #     labels=metadata[:, 1],
    #     percentages=percentages,
    #     embed_kwargs=embed_kwargs, 
    #     lr_kwargs=lr_kwargs,
    #     seed=42
    # )

    results_normal, betas_normal = link_lr_with_expected_fringe_degree_auc(
        adj_matrix=adj_matrix,
        core_indices=random_core_indices,
        fringe_indices=random_fringe_indices,
        y_core=random_gender_core,
        y_fringe=random_gender_fringe,
        percentages=percentages,
        lr_kwargs=lr_kwargs,
        seed=42
    )


    results_normal_2, betas_normal_2 = link_lr_with_expected_fringe_degree_auc(
        adj_matrix=adj_matrix,
        core_indices=random_core_indices,
        fringe_indices=random_fringe_indices,
        y_core=random_gender_core,
        y_fringe=random_gender_fringe,
        percentages=percentages,
        lr_kwargs=lr_kwargs,
        seed=123
    )

    results_normal_core, betas_normal_core = link_logistic_regression_core_only_auc(
        adj_matrix=adj_matrix,
        core_indices=random_core_indices,
        fringe_indices=random_fringe_indices,
        y_core=random_gender_core,
        y_fringe=random_gender_fringe,
        percentages=percentages,
        lr_kwargs=lr_kwargs,
        seed=42
    )

    # plot_beta_vectors(betas_normal, "Yale_IID_Sample")
    # plot_beta_comparison(betas_normal, betas_normal_core, "Yale_IID_Sample")
    plot_beta_comparison(betas_normal_2, betas_normal, "Yale_IID_Sample_Train_Twice")

    sizes = (core_size, fringe_size, normal_fringe_size)
    return {'multi_dorm': results_multi, 'normal': results_normal, 'sizes': sizes, 'stats': stats_string, 'normal_stats': normal_stats_string}


def run_experiments_node2vec(adj_matrix, metadata, chosen_dorms_list, percentages, seed=None):
    """
    Run the experiment for each set of dormIDs in chosen_dorms_list.
    For each set, compare the multi-dorm core experiment with the normal graph experiment.
    """
    experiment_results = {}
    for dorm_ids in chosen_dorms_list:
        print("\n===================================")
        print(f"Running experiment for dormIDs: {dorm_ids}")
        results = perform_experiment_for_multi_dorm_core_node2vec(adj_matrix, metadata, dorm_ids, percentages, seed=seed)
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


#########################################
# Example Usage for Hyperparameter Search
#########################################

if __name__ == '__main__':


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
            experiment_results = run_experiments_node2vec(adj_matrix, metadata, chosen_dorms_list, percentages, seed=123)
            # plot_experiment_results_barline(experiment_results, title_prefix="Performance Comparison (node2vec)", save_filename_prefix=f"../figures/exp_results_{tag}_auc_node2vec_regularizer_dynamic")
            # dorm_ids = [[np.uint(31), np.uint(32)]]
            # multi_dorm_graph, multi_dorm_core_indices, multi_dorm_fringe_indices = create_multi_dorm_core_fringe_graph(adj_matrix, metadata, dorm_ids)
            # multi_gender_core, multi_gender_fringe = prepare_core_fringe_attributes(metadata, multi_dorm_core_indices, multi_dorm_fringe_indices)
            # n = adj_matrix.shape[0]
            # k = 976  # sample same number of nodes as the multi-dorm core size.
            # # if seed is not None:
            # #     np.random.seed(seed)
            # np.random.seed(42)
            # random_core_indices = np.random.choice(np.arange(n), size=k, replace=False)
            # # Define fringe as all other nodes (the complement).
            # all_nodes = np.arange(n)
            # random_fringe_indices = np.setdiff1d(all_nodes, random_core_indices)
            # # print(f"=== Multi-Dorm Core Experiment using dormIDs: {dorm_ids} ===")
            # # print(f"Multi-dorm core size: {len(multi_dorm_core_indices)}")
            # # Define a parameter grid for Node2Vec hyperparameters.
            # # Define your grid
            # param_grid = {
            #     'dimensions': [32, 64],
            #     'walk_length': [10, 20],
            #     'num_walks': [5, 10],
            #     'window_size': [5, 10],
            #     'p': [0.5, 1.0],
            #     'q': [1.0, 2.0],
            #     'alpha': [0.01, 0.025],
            # }
            # lr_C_list = [0.1, 1.0, 10.0]
            
            # best_params, best_acc, results = hyperparameter_search_node2vec_gensim(
            #     adj_matrix=adj_matrix,
            #     core_indices=random_core_indices,
            #     fringe_indices=random_fringe_indices,
            #     labels=metadata[:, 1],
            #     param_grid=param_grid,
            #     lr_C_list=lr_C_list,
            #     seed=42
            # )
            
            # # Optionally, print all search results.
            # for params, acc in results:
            #     print("Params:", params, "-> Accuracy:", acc)
