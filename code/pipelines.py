# import matplotlib.pyplot as plt 
import itertools
from methods import *
from data_preprocessing import *
from plotter import *


def finetuneLR():
    file_ext = '.mat'
    best_auc = -1
    best_C = None
    best_solver = None
    best_beta = None
    best_acc = None
    best_model_type = None  # core_only or not

    # Try a range of C values (regularization strengths) and solvers
    C_values = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000, 5000, 10000]
    solvers = ['liblinear', 'lbfgs', 'saga', 'newton-cg', 'sag']
    for f in listdir(fb_code_path):
        if f.endswith(file_ext):
            print(f)
            # adj_matrix, metadata = parse_fb100_mat_file(path_join(fb_code_path, f))
            # chosen_dorms_list = [np.uint(31), np.uint(32)]
            # adj_matrix, core_indices, fringe_indices = create_multi_dorm_core_fringe_graph(adj_matrix, metadata, chosen_dorms_list)
            adj_matrix, core_indices, fringe_indices, metadata = sbm_manual_core_fringe(1000, 400, 0.15, 0.1, seed=999)
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



def dorm_core_pipeline():
    file_ext = '.mat'
    auc_scores = {
        'cc' : [],
        'cf' : [],
        'cfed' : [],
        'cfed_naive' : [],  
        'node2vec' : []
    }
    acc_scores = {
        'cc' : [],
        'cf' : [],
        'cfed' : [],
        'cfed_naive' : [],
        'node2vec' : []
    }

    for f in listdir(fb_code_path):
        if f.endswith(file_ext):
            print(f)
            adj_matrix, metadata = parse_fb100_mat_file(path_join(fb_code_path, f))
            chosen_dorms_list = [[np.uint(31), np.uint(32)]]
            adj_matrix, core_indices, fringe_indices = create_multi_dorm_core_fringe_graph(adj_matrix, metadata, chosen_dorms_list)
            percentages = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            lr_kwargs = {'C': 0.01, 'solver': 'newton-cg', 'max_iter': 1000}
            embed_kwargs={'dimensions': 32, 'walk_length': 40, 'num_walks': 5, 'window_size': 5, 'p': 0.5, 'q': 1.0, 'alpha': 0.025}
            for p in percentages:
                labelled_core_indices = np.random.choice(core_indices, size=int(p * len(core_indices)), replace=False)
                beta_core_only_p, acc_cc, auc_cc = link_logistic_regression_pipeline(adj_matrix, labelled_core_indices, fringe_indices, metadata, core_only=True, lr_kwargs=lr_kwargs)
                beta_core_fringe_p, acc_cf, auc_cf = link_logistic_regression_pipeline(adj_matrix, labelled_core_indices, fringe_indices, metadata, core_only=False, lr_kwargs=lr_kwargs)    
                beta_cfed, acc_cfed, auc_cfed = link_logistic_regression_pipeline(adj_matrix, labelled_core_indices, fringe_indices, metadata, core_only=False, lr_kwargs=lr_kwargs, expected_degree=True)
                beta_cfed_naive, acc_cfed_naive, auc_cfed_naive = link_logistic_regression_pipeline(adj_matrix, labelled_core_indices, fringe_indices, metadata, core_only=False, lr_kwargs=lr_kwargs, expected_degree=True, naive_degree=True)
                beta_node2vec, acc_node2vec, auc_node2vec = node2vec_logistic_regression_pipeline(adj_matrix, labelled_core_indices, fringe_indices, metadata, core_only=False, lr_kwargs=lr_kwargs, embed_kwargs=embed_kwargs)
                # print("Correlation:", np.corrcoef(beta_core_only_p, beta_core_fringe_p[labelled_core_indices])[0, 1])
                # padded_beta_core_only = np.full_like(beta_core_fringe_p, np.nan)
                # padded_beta_core_only[labelled_core_indices] = beta_core_only_p
                # plot_beta_comparison(beta_core_fringe_p, padded_beta_core_only, f"Yale_31_32_pipeline_padded_{p}")
                auc_scores['cc'].append(auc_cc)
                auc_scores['cf'].append(auc_cf)
                auc_scores['cfed'].append(auc_cfed)
                auc_scores['cfed_naive'].append(auc_cfed_naive)
                auc_scores['node2vec'].append(auc_node2vec) 
                acc_scores['cc'].append(acc_cc)
                acc_scores['cf'].append(acc_cf)
                acc_scores['cfed'].append(acc_cfed)
                acc_scores['cfed_naive'].append(acc_cfed_naive) 
                acc_scores['node2vec'].append(acc_node2vec)
            # beta_core_only = link_logistic_regression_pipeline(adj_matrix, core_indices, fringe_indices, metadata, core_only=True)
            # beta_core_fringe = link_logistic_regression_pipeline(adj_matrix, core_indices, fringe_indices, metadata, core_only=False)
            # print("Correlation:", np.corrcoef(beta_core_only, beta_core_fringe[core_indices])[0, 1])
            # plot_beta_comparison(beta_core_only, beta_core_fringe[core_indices], "Yale_31_32_pipeline")
    # plot_auc(auc_scores, acc_scores, percentages, "Yale_31_32")


def iid_pipeline():
    file_ext = '.mat'
    auc_scores = {
        'cc' : [],
        'cf' : [],
        'cfed' : [],
        'cfed_naive' : [],
        'node2vec' : []
    }
    acc_scores = {  
        'cc' : [],
        'cf' : [],
        'cfed' : [],
        'cfed_naive' : [],
        'node2vec' : []
    }
    
    for f in listdir(fb_code_path):
        if f.endswith(file_ext):
            print(f)
            # adj_matrix, metadata = parse_fb100_mat_file(path_join(fb_code_path, f))
            adj_matrix, metadata = sbm_gender_homophily_adj_and_metadata(300, 300, 0.15, 0.1, seed=42)
            assortativity = nx.degree_assortativity_coefficient(nx.from_numpy_array(adj_matrix))
            print(f"Assortativity: {assortativity}")
            # break
            core_fringe_adj, core_indices, fringe_indices = create_iid_core_fringe_graph(adj_matrix, 300, seed=42)
            percentages = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            lr_kwargs = {'C': 0.01, 'solver': 'newton-cg', 'max_iter': 1000}
            embed_kwargs={'dimensions': 32, 'walk_length': 40, 'num_walks': 5, 'window_size': 5, 'p': 0.5, 'q': 1.0, 'alpha': 0.025}
            for p in percentages:
                labelled_core_indices = np.random.choice(core_indices, size=int(p * len(core_indices)), replace=False)
                beta_core_only, acc_core_only, auc_core_only = link_logistic_regression_pipeline(core_fringe_adj, labelled_core_indices, fringe_indices, metadata, core_only=True, lr_kwargs=lr_kwargs)
                beta_core_fringe, acc_core_fringe, auc_core_fringe = link_logistic_regression_pipeline(core_fringe_adj, labelled_core_indices, fringe_indices, metadata, core_only=False, lr_kwargs=lr_kwargs)
                beta_cfed, acc_cfed, auc_cfed = link_logistic_regression_pipeline(core_fringe_adj, labelled_core_indices, fringe_indices, metadata, core_only=False, lr_kwargs=lr_kwargs, expected_degree=True, iid_core=True)
                beta_cfed_naive, acc_cfed_naive, auc_cfed_naive = link_logistic_regression_pipeline(core_fringe_adj, labelled_core_indices, fringe_indices, metadata, core_only=False, lr_kwargs=lr_kwargs, expected_degree=True, iid_core=True, naive_degree=True)
                beta_node2vec, acc_node2vec, auc_node2vec = node2vec_logistic_regression_pipeline(core_fringe_adj, labelled_core_indices, fringe_indices, metadata, core_only=False, lr_kwargs=lr_kwargs, embed_kwargs=embed_kwargs)
                # print("Correlation:", np.corrcoef(beta_core_only, beta_core_fringe[labelled_core_indices])[0, 1])
                # padded_beta_core_only = np.full_like(beta_core_fringe, np.nan)
                # padded_beta_core_only[labelled_core_indices] = beta_core_only
                # plot_beta_comparison(beta_core_fringe, padded_beta_core_only, f"SBM_gender_homophily_iid_pipeline_padded_{p}")
                auc_scores['cc'].append(auc_core_only)
                auc_scores['cf'].append(auc_core_fringe)
                auc_scores['cfed'].append(auc_cfed)
                auc_scores['cfed_naive'].append(auc_cfed_naive)
                auc_scores['node2vec'].append(auc_node2vec)
                acc_scores['cc'].append(acc_core_only)
                acc_scores['cf'].append(acc_core_fringe)
                acc_scores['cfed'].append(acc_cfed)
                acc_scores['cfed_naive'].append(acc_cfed_naive)
                acc_scores['node2vec'].append(acc_node2vec)
    # plot_auc(auc_scores, acc_scores, percentages, f"SBM_gender_homophily_iid_0.15_0.1")



def sbm_pipeline(n_runs=2):
    file_ext = '.mat'
    percentages = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    seeds = [123, 345, 678, 910, 112]
    n_steps = len(percentages)
    auc_scores = { 'cc': [], 'cf': [], 'cfed_true': [], 'random': []}
    auc_cis = { 'cc': [], 'cf': [], 'cfed_true': [],  'random' : []}
    acc_scores = { 'cc': [], 'cf': [], 'cfed_true': [],  'random' : [] }
    for run in range(n_runs):
        print(f"SBM pipeline run {run+1}/{n_runs}")
        for f in listdir(fb_code_path):
            if f.endswith(file_ext):
                print(f)
                # seed = seeds[run]
                from datetime import datetime
                seed = random.seed(datetime.now().timestamp())
                sbm_adj_matrix, metadata = sbm_gender_homophily_adj_and_metadata(500, 500, 0.15, 0.1, seed=seed)
                adj_matrix, core_indices, fringe_indices, ff_true = create_iid_core_fringe_graph(sbm_adj_matrix, 300, seed=seed, ff=True)
                # adj_matrix, core_indices, fringe_indices, metadata = sbm_manual_core_fringe(1000, 400, 0.15, 0.1, seed=seed)
                lr_kwargs = {'C': 100, 'solver': 'liblinear', 'max_iter': 1000}
                for idx, p in enumerate(percentages):
                    labelled_core_indices = np.random.choice(core_indices, size=int(p * len(core_indices)), replace=False)
                    _, acc_core_only, auc_core_only, ci_cc = link_logistic_regression_pipeline(adj_matrix, labelled_core_indices, fringe_indices, metadata, core_only=True, lr_kwargs=lr_kwargs, return_auc_ci=True)
                    _, acc_core_fringe, auc_core_fringe, ci_cf = link_logistic_regression_pipeline(adj_matrix, labelled_core_indices, fringe_indices, metadata, core_only=False, lr_kwargs=lr_kwargs, return_auc_ci=True)
                    _, acc_cfed, auc_cfed, ci_cfed = link_logistic_regression_pipeline(adj_matrix, labelled_core_indices, fringe_indices, metadata, core_only=False, lr_kwargs=lr_kwargs, expected_degree=False, ff=True, ff_value=ff_true, p_core_fringe=0.15, p_fringe_fringe=0.0, return_auc_ci=True)
                    # _, acc_cfed_naive, auc_cfed_naive, ci_cfed_naive = link_logistic_regression_pipeline(adj_matrix, labelled_core_indices, fringe_indices, metadata, core_only=False, lr_kwargs=lr_kwargs, expected_degree=True, iid_core=True, p_core_fringe=0.15, p_fringe_fringe=0.0, naive_degree=True, return_auc_ci=True)
                    
                    auc_random, ci_random = random_guesser(fringe_indices, metadata, seed=seed)
                    auc_scores['cc'].append(auc_core_only)
                    auc_scores['cf'].append(auc_core_fringe)
                    auc_scores['cfed_true'].append(auc_cfed)
                    # auc_scores['cfed_naive'].append(auc_cfed_naive)
                    auc_scores['random'].append(auc_random)
                    auc_cis['cc'].append(ci_cc)
                    auc_cis['cf'].append(ci_cf)
                    auc_cis['cfed_true'].append(ci_cfed)
                    # auc_cis['cfed_naive'].append(ci_cfed_naive)
                    auc_cis['random'].append(ci_random)
                    acc_scores['cc'].append(acc_core_only)
                    acc_scores['cf'].append(acc_core_fringe)
                    acc_scores['cfed_true'].append(acc_cfed)
                    # acc_scores['cfed_naive'].append(acc_cfed_naive)
    # Convert lists to arrays for easier reshaping
    for key in auc_scores:
        auc_scores[key] = np.array(auc_scores[key]).reshape(n_runs, len(percentages))
        auc_cis[key] = np.array(auc_cis[key]).reshape(n_runs, len(percentages), 2)

    # Average AUCs and CIs over runs
    auc_means = {k: auc_scores[k].mean(axis=0) for k in auc_scores}
    auc_cis_means = {
        k: np.stack([
            auc_cis[k][:, :, 0].mean(axis=0),  # lower
            auc_cis[k][:, :, 1].mean(axis=0)   # upper
        ], axis=1)
        for k in auc_cis
    }

    # Prepare for plotting
    auc_scores_plot = {k: auc_means[k] for k in auc_means}
    auc_cis_plot = {k: list(zip(auc_cis_means[k][:, 0], auc_cis_means[k][:, 1])) for k in auc_cis_means}

    # Reshape accuracy scores similarly
    for key in acc_scores:
        if key != 'random':  # Skip random as it's not used in accuracy plots
            acc_scores[key] = np.array(acc_scores[key]).reshape(n_runs, len(percentages)).mean(axis=0)

    plot_auc_with_ci(auc_scores_plot, auc_cis_plot, percentages, f"SBM_pipeline_increased_avg{n_runs}_0.15_0.1_gender_v5")
    plot_acc(acc_scores, percentages, f"SBM_gender_0.15_0.1_v5")



def fringe_inclusion_pipeline_and_plot(n_core=150, n_fringe=50, p_core_core=0.8, p_core_fringe=0.1, p_fringe_fringe=0.0, n_steps=5, lr_kwargs=None, tag="SBM_fringe_inclusion"):
    """
    For each file in fb_code_path, generate a new SBM core-fringe graph.
    For each step, train only on core_indices (not including any fringe in training), but incrementally add more core-fringe edges to the adjacency (i.e., more of the core-fringe block is revealed).
    Always test on the full fringe set.
    Plots accuracy and AUC for Core-Fringe and Core-Fringe-Expected Degree models, with 95% CI.
    """
    import matplotlib.pyplot as plt
    if lr_kwargs is None:
        lr_kwargs = {'C': 0.01, 'solver': 'newton-cg', 'max_iter': 1000}
    auc_scores = {'cf': [], 'cfed': [], 'cc': [], 'random': []}
    auc_cis = {'cf': [], 'cfed': [], 'cc': [], 'random': []}
    acc_scores = {'cf': [], 'cfed': [], 'cc': []}
    fringe_pct = []
    file_ext = '.mat'
    for f in listdir(fb_code_path):
        if f.endswith(file_ext):
            print(f)
            from datetime import datetime
            seed = random.seed(datetime.now().timestamp())
            sbm_adj_matrix, metadata = sbm_gender_homophily_adj_and_metadata(500, 500, 0.15, 0.1, seed=seed)
            adj_matrix, core_indices, fringe_indices = create_iid_core_fringe_graph(sbm_adj_matrix, 300, seed=seed)
            n_fringe = len(fringe_indices)
            step_size = n_fringe // n_steps
            for i in range(0, n_steps + 1):
                n_train_fringe = i * step_size if i < n_steps else n_fringe
                revealed_fringe = fringe_indices[:n_train_fringe]
                train_indices = np.concatenate([core_indices, revealed_fringe])

                # Core-Fringe model
                _, acc_cf, auc_cf, ci_cf = link_logistic_regression_pipeline(
                    adj_matrix, train_indices, fringe_indices, metadata, core_only=False, lr_kwargs=lr_kwargs, return_auc_ci=True,
                )

                # Core-Fringe Expected Degree model
                _, acc_cfed, auc_cfed, ci_cfed = link_logistic_regression_pipeline(
                    adj_matrix, train_indices, fringe_indices, metadata, core_only=False, lr_kwargs=lr_kwargs,
                    expected_degree=True, sbm=True, p_core_fringe=p_core_fringe, p_fringe_fringe=p_fringe_fringe, return_auc_ci=True
                )

                _, acc_cc, auc_cc, ci_cc = link_logistic_regression_pipeline(
                    adj_matrix, core_indices, fringe_indices, metadata, core_only=True, lr_kwargs=lr_kwargs, return_auc_ci=True
                )

                auc_random, ci_random = random_guesser(fringe_indices, metadata, seed=seed)

                auc_scores['cf'].append(auc_cf)
                auc_scores['cfed'].append(auc_cfed)
                auc_scores['cc'].append(auc_cc)
                auc_scores['random'].append(auc_random)
                auc_cis['cf'].append(ci_cf)
                auc_cis['cfed'].append(ci_cfed)
                auc_cis['cc'].append(ci_cc)
                auc_cis['random'].append(ci_random)
                acc_scores['cf'].append(acc_cf)
                acc_scores['cfed'].append(acc_cfed)
                acc_scores['cc'].append(acc_cc)
                fringe_pct.append(n_train_fringe / n_fringe)
                print(f"Step {i}: Revealed {n_train_fringe} fringe, tested on {n_fringe} fringe")

            # Add random guesser baseline (same for all steps)
            # auc_random, ci_random = random_guesser(fringe_indices, metadata)
            # auc_scores['random'] = [auc_random] * len(fringe_pct)
            # auc_cis['random'] = [ci_random] * len(fringe_pct)

    # Plotting with CI
    plt.figure(figsize=(10, 5))
    methods = ['cc', 'cf', 'cfed', 'random']
    for method in methods:
        aucs = np.array(auc_scores[method])
        lowers = np.array([ci[0] for ci in auc_cis[method]])
        uppers = np.array([ci[1] for ci in auc_cis[method]])
        yerr = np.vstack([aucs - lowers, uppers - aucs])
        plt.errorbar(fringe_pct, aucs, yerr=yerr, label=method, marker='o', capsize=4)
    plt.xticks(fringe_pct)
    plt.xlabel('Fraction of Fringe Nodes with Revealed Edges')
    plt.ylabel('AUC')
    plt.title(f'AUC vs. Fringe Edge Reveal Fraction with 95% CI ({tag})')
    plt.legend()
    plt.savefig(f"../figures/{tag}_auc_comparison_gender_0.15_0.1_CI_v2.png")
    plt.close()


def sbm_homophily_sweep_pipeline(n_runs=5, n_core=1000, n_fringe=400, p_cc_values=None, p_cf_values=None, p_ff=0.0, tag="SBM_homophily_sweep_gender_homophily_wo_random"):
    """
    For each value of P_CC and P_CF (with P_CC/P_CF > 1), generate a SBM core-fringe graph.
    Always use 100% of the core as labelled for training.
    For each (P_CC, P_CF) pair, run the pipeline and store the average AUC and accuracy over n_runs.
    Plot AUC and accuracy as a function of (P_CC, P_CF) tuple for all methods.
    """
    import matplotlib.pyplot as plt
    if p_cc_values is None:
        p_cc_values = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    if p_cf_values is None:
        p_cf_values = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    seeds = [123, 345, 678, 910, 112]
    method_keys = ['cc', 'cf', 'cfed']
    auc_dict = {k: [] for k in method_keys}
    auc_cis = {k: [] for k in method_keys}
    acc_dict = {k: [] for k in method_keys if k != 'random'}
    tuple_list = []
    p_cf = 0.15
    ratios = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
    # for i, p_cc in enumerate(p_cc_values):
    for j, r in enumerate(ratios):
        # if p_cc / p_cf <= 1:
        #     continue
        p_cc = r * p_cf
        auc_runs = {k: [] for k in method_keys}
        auc_ci_runs = {k: [] for k in method_keys}
        acc_runs = {k: [] for k in method_keys if k != 'random'}
        for run in range(n_runs):
            # seed = seeds[run % len(seeds)]
            from datetime import datetime
            seed = random.seed(datetime.now().timestamp())
            # adj_matrix, core_indices, fringe_indices, metadata = sbm_manual_core_fringe(n_core, n_fringe, p_cc, p_cf, seed=seed)
            sbm_adj_matrix, metadata = sbm_gender_homophily_adj_and_metadata(500, 500, p_cc, p_cf, seed=seed)
            adj_matrix, core_indices, fringe_indices = create_iid_core_fringe_graph(sbm_adj_matrix, 300, seed=seed)
            lr_kwargs = {'C': 100, 'solver': 'liblinear', 'max_iter': 1000}
            train_indices = core_indices
            # Core-Core
            _, acc_cc, auc_cc, ci_cc = link_logistic_regression_pipeline(adj_matrix, train_indices, fringe_indices, metadata, core_only=True, lr_kwargs=lr_kwargs, return_auc_ci=True)
            # Core-Fringe
            _, acc_cf, auc_cf, ci_cf = link_logistic_regression_pipeline(adj_matrix, train_indices, fringe_indices, metadata, core_only=False, lr_kwargs=lr_kwargs, return_auc_ci=True)
            # Core-Fringe Expected Degree
            _, acc_cfed, auc_cfed, ci_cfed = link_logistic_regression_pipeline(adj_matrix, train_indices, fringe_indices, metadata, core_only=False, lr_kwargs=lr_kwargs, expected_degree=True, sbm=True, p_core_fringe=p_cf, p_fringe_fringe=p_ff, return_auc_ci=True)
            # Core-Fringe Naive Expected Degree
            # _, acc_cfed_naive, auc_cfed_naive, ci_cfed_naive = link_logistic_regression_pipeline(adj_matrix, train_indices, fringe_indices, metadata, core_only=False, lr_kwargs=lr_kwargs, expected_degree=True, sbm=True, p_core_fringe=p_cf, p_fringe_fringe=p_ff, naive_degree=True, return_auc_ci=True)
            # Random guesser
            # auc_random, ci_random = random_guesser(fringe_indices, metadata, seed=seed)
            auc_runs['cc'].append(auc_cc)
            auc_runs['cf'].append(auc_cf)
            auc_runs['cfed'].append(auc_cfed)
            # auc_runs['cfed_naive'].append(auc_cfed_naive)
            # auc_runs['random'].append(auc_random)
            auc_ci_runs['cc'].append(ci_cc)
            auc_ci_runs['cf'].append(ci_cf)
            auc_ci_runs['cfed'].append(ci_cfed)
            # auc_ci_runs['cfed_naive'].append(ci_cfed_naive)
            # auc_ci_runs['random'].append(ci_random)
            acc_runs['cc'].append(acc_cc)
            acc_runs['cf'].append(acc_cf)
            acc_runs['cfed'].append(acc_cfed)
            # acc_runs['cfed_naive'].append(acc_cfed_naive)
        for k in method_keys:
            auc_dict[k].append(np.mean(auc_runs[k]))
            lowers = [ci[0] for ci in auc_ci_runs[k]]
            uppers = [ci[1] for ci in auc_ci_runs[k]]
            auc_cis[k].append((np.mean(lowers), np.mean(uppers)))
        for k in acc_dict:
            acc_dict[k].append(np.mean(acc_runs[k]))
        tuple_list.append((p_cc, p_cf))
    # Plotting
    xtick_labels = [f"({p_cc:.2f},{p_cf:.2f})" for p_cc, p_cf in tuple_list]
    x = np.arange(len(tuple_list))
    plt.figure(figsize=(24, 10))
    for k in method_keys:
        aucs = np.array(auc_dict[k])
        lowers = np.array([ci[0] for ci in auc_cis[k]])
        uppers = np.array([ci[1] for ci in auc_cis[k]])
        yerr = np.vstack([aucs - lowers, uppers - aucs])
        plt.errorbar(x, aucs, yerr=yerr, label=k.replace('_', '-').title(), marker='o', capsize=4)
    plt.xlabel('P_IN, P_OUT')
    plt.ylabel('AUC')
    plt.title(f'AUC vs. (P_IN, P_OUT) with 95% CI ({tag})')
    plt.xticks(x, xtick_labels, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"../figures/{tag}_auc_vs_tuple_CI.png")
    plt.close()
    # (You can keep the accuracy plot as before if desired)
    return auc_dict, acc_dict, tuple_list



def hyperparameter_search_node2vec():
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
      - seed: random seedn for reproducibility
    
    Returns:
      - best_combo: dict of best embedding params + best C
      - best_acc: highest fringe accuracy
      - all_results: list of tuples (combo_dict, accuracy)
    """

    from datetime import datetime
    seed = random.seed(datetime.now().timestamp())
    param_grid = {
        'dimensions': [32, 64],
        'walk_length': [10, 20],
        'num_walks': [5, 10],
        'window_size': [5, 10],
        'p': [0.5, 1.0],
        'q': [1.0, 2.0],
        'alpha': [0.01, 0.025],
    }
    lr_C_list = [0.1, 1.0, 10.0]
    sbm_adj_matrix, metadata = sbm_gender_homophily_adj_and_metadata(500, 500, 0.15, 0.1, seed=seed)
    adj_matrix, core_indices, fringe_indices = create_iid_core_fringe_graph(sbm_adj_matrix, 300, seed=seed)
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
                beta, acc, auc = node2vec_logistic_regression_pipeline(
                    adj_matrix=adj_matrix,
                    core_indices=core_indices, 
                    fringe_indices=fringe_indices,
                    metadata=metadata,
                    lr_kwargs=lr_kwargs,
                    embed_kwargs=embed_kwargs,
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



def test_fringe_prediction_accuracy():
    from datetime import datetime
    import matplotlib.pyplot as plt
    
    p_out = 0.15
    ratios = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
    
    # Store metrics for each ratio
    all_metrics = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1_score': [],
        'true_density': [],
        'predicted_density': [],
        'total_edges_true': [],
        'total_edges_predicted': []
    }
    
    for r in ratios:
        p_in = r * p_out
        seed = random.seed(datetime.now().timestamp())
        sbm_adj_matrix, metadata = sbm_gender_homophily_adj_and_metadata(500, 500, p_in, p_out, seed=seed)
        adj_matrix, core_indices, fringe_indices, ff_true = create_iid_core_fringe_graph(sbm_adj_matrix, 300, seed=seed, ff=True)
        adj_imputed = expected_degree_imputation(adj_matrix, core_indices, fringe_indices, 500, 500, p_in, p_out, metadata)
        ff_predicted = adj_imputed[fringe_indices, :][:, fringe_indices]
        
        # Compare predictions
        metrics = compare_fringe_fringe_predictions(ff_true, ff_predicted)
        
        # Store metrics
        for key in all_metrics:
            all_metrics[key].append(metrics[key])
        
        # Print results for this ratio
        print(f"\nRatio p_in/p_out = {r:.1f}")
        print(f"Accuracy: {metrics['accuracy']:.3f}")
        print(f"Precision: {metrics['precision']:.3f}")
        print(f"Recall: {metrics['recall']:.3f}")
        print(f"F1 Score: {metrics['f1_score']:.3f}")
        print(f"True density: {metrics['true_density']:.3f}")
        print(f"Predicted density: {metrics['predicted_density']:.3f}")
    
    # Create plots
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Performance Metrics
    plt.subplot(2, 1, 1)
    plt.plot(ratios, all_metrics['accuracy'], 'o-', label='Accuracy')
    plt.plot(ratios, all_metrics['precision'], 's-', label='Precision')
    plt.plot(ratios, all_metrics['recall'], '^-', label='Recall')
    plt.plot(ratios, all_metrics['f1_score'], 'd-', label='F1 Score')
    plt.xlabel('Ratio p_in/p_out')
    plt.ylabel('Score')
    plt.title('Prediction Performance Metrics vs. p_in/p_out Ratio')
    plt.legend()
    plt.grid(True)
    
    # Plot 2: Density Comparison
    plt.subplot(2, 1, 2)
    plt.plot(ratios, all_metrics['true_density'], 'o-', label='True Density')
    plt.plot(ratios, all_metrics['predicted_density'], 's-', label='Predicted Density')
    plt.xlabel('Ratio p_in/p_out')
    plt.ylabel('Density')
    plt.title('Edge Density Comparison vs. p_in/p_out Ratio')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('../figures/fringe_prediction_metrics_vs_ratio_class_info.png')
    plt.close()
    
    # Create a separate plot for edge counts
    plt.figure(figsize=(10, 6))
    plt.plot(ratios, all_metrics['total_edges_true'], 'o-', label='True Edges')
    plt.plot(ratios, all_metrics['total_edges_predicted'], 's-', label='Predicted Edges')
    plt.xlabel('Ratio p_in/p_out')
    plt.ylabel('Number of Edges')
    plt.title('Edge Counts vs. p_in/p_out Ratio')
    plt.legend()
    plt.grid(True)
    plt.savefig('../figures/fringe_prediction_edge_counts_vs_ratio_class_info.png')
    plt.close()





if __name__ == '__main__':
    # sbm_pipeline()
    # test_fringe_prediction_accuracy()
    # hyperparameter_search_node2vec()
    # sbm_homophily_sweep_pipeline()
    fringe_inclusion_pipeline_and_plot()