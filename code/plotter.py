import matplotlib.pyplot as plt 
import numpy as np

def plot_acc(acc_scores, percentages, tag):
    
    plt.figure(figsize=(10, 5))
    plt.plot(percentages, acc_scores['cc'], label='Core-Core', marker='o')
    plt.plot(percentages, acc_scores['cf'], label='Core-Fringe', marker='o')
    plt.plot(percentages, acc_scores['cfed_true'], label='Core-Fringe (True Fringe Degree)', marker='o')
    # plt.plot(percentages, acc_scores['cfed_naive'], label='Core-Fringe (Naive Expected Degree)', marker='o')
    # plt.plot(percentages, acc_scores['node2vec'], label='Node2Vec', marker='o')
    plt.xlabel('Percentage of Core Nodes Used for Training')
    plt.xticks(percentages)
    plt.ylabel('Accuracy')
    plt.title(f'Accuracy Comparison for {tag}')
    plt.legend()
    plt.savefig(f"../figures/{tag}_acc_comparison.png")
    plt.close()


def plot_auc_with_ci(auc_scores, auc_cis, percentages, tag):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    methods = list(auc_scores.keys())
    for method in methods:
        aucs = np.array(auc_scores[method])
        lowers = np.array([ci[0] for ci in auc_cis[method]])
        uppers = np.array([ci[1] for ci in auc_cis[method]])
        yerr = np.vstack([aucs - lowers, uppers - aucs])
        plt.errorbar(percentages, aucs, yerr=yerr, label=method, marker='o', capsize=4)
    plt.xlabel('Percentage of Core Nodes Used for Training')
    plt.xticks(percentages)
    plt.ylabel('AUC')
    plt.title(f'AUC Comparison with 95% CI for {tag}')
    plt.legend()
    plt.savefig(f"../figures/{tag}_auc_comparison_CI.png")
    plt.close()

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