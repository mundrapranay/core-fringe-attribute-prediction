import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score


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


def link_logistic_regression_pipeline(adj_matrix, core_indices, fringe_indices, metadata, core_only=False, lr_kwargs=None, expected_degree=False):
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