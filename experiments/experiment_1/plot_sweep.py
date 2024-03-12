import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json

OUTPUT_DIR = 'results/plots'


from plot_utils import lighten_color, change_saturation, get_fancy_bbox

def load_data(path):
    return json.load(open(path))

def calculate_statistics(datasets):
    """Calculate means, sample sizes, and standard errors for a list of datasets."""
    means = [np.mean(dataset) for dataset in datasets]
    ns = [len(dataset) for dataset in datasets]
    print(ns)
    errors = [1.96 * np.std(dataset, ddof=1) / np.sqrt(n) for dataset, n in zip(datasets, ns)]
    return means, ns, errors

def plot_results(temperatures, means_ft, errors_ft, labels, title, filename):
    """Plot the results with error bars for each feature type (ft) and save to PDF."""
    # plt.rcParams["font.family"] = "Avenir"
    plt.rcParams["font.size"] = 28
    palette = sns.color_palette('colorblind')
    fig, ax = plt.subplots(figsize=(10, 10))

    for i, (means, errors) in enumerate(zip(means_ft, errors_ft)):
        ax.errorbar(temperatures, means, yerr=errors, fmt='o-', color=palette[i], label=labels[i])

    ax.set_ylabel('Win Rate')
    ax.set_xlabel('Iteration')
    ax.set_xticks(temperatures)
    ax.set_xticklabels(map(str, temperatures))
    ax.set_yticks([0.2,  0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    ax.set_yticklabels([0.2,  0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    ax.set_ylim(0.175, .925)  
    ax.set_xlim(min(temperatures) - 0.05, max(temperatures) + 0.05)
    ax.axhline(y=0.5, color='lightgrey', linestyle='--', linewidth=2)
    ax.legend(loc='upper left', ncol=2, fontsize=8, frameon=False, bbox_to_anchor=(0.25, 0.95), prop={'size': 16})

    sns.despine()
    plt.title(title)
    plt.tight_layout()
    
    plt.savefig(f'{OUTPUT_DIR}/{filename}_grid.pdf')
    plt.savefig(f'{OUTPUT_DIR}/{filename}_grid.png')
    plt.close()  # Close the plot to prevent it from showing inline if using Jupyter

def main(): 
    # beta 0.5
    helpful_beta_05 = [
        "results/responses/sweep/win_rates/typo-iteration-0vsbase-lr-1e-6-beta-0.5-temperature-0.0-helpful.json",
        "results/responses/sweep/win_rates/typo-iteration-1vs0-lr-1e-6-beta-0.5-temperature-0.0-helpful.json",
        "results/responses/sweep/win_rates/typo-iteration-2vs1-lr-1e-6-beta-0.5-temperature-0.0-helpful.json",
    ]
    
    harmless_beta_05 = [
        "results/responses/sweep/win_rates/typo-iteration-0vsbase-lr-1e-6-beta-0.5-temperature-0.0-harmless.json",
        "results/responses/sweep/win_rates/typo-iteration-1vs0-lr-1e-6-beta-0.5-temperature-0.0-harmless.json",
        "results/responses/sweep/win_rates/typo-iteration-2vs1-lr-1e-6-beta-0.5-temperature-0.0-harmless.json",
    ]
     
    # beta 1.0
    helpful_beta_1 = [
        "results/responses/sweep/win_rates/typo-iteration-0vsbase-lr-1e-6-beta-1.0-temperature-0.0-helpful.json",
        "results/responses/sweep/win_rates/typo-iteration-1vs0-lr-1e-6-beta-1.0-temperature-0.0-helpful.json",
        "results/responses/sweep/win_rates/typo-iteration-2vs1-lr-1e-6-beta-1.0-temperature-0.0-helpful.json",
    ]
    
    harmless_beta_1 = [
        "results/responses/sweep/win_rates/typo-iteration-0vsbase-lr-1e-6-beta-1.0-temperature-0.0-harmless.json",
        "results/responses/sweep/win_rates/typo-iteration-1vs0-lr-1e-6-beta-1.0-temperature-0.0-harmless.json",
        "results/responses/sweep/win_rates/typo-iteration-2vs1-lr-1e-6-beta-1.0-temperature-0.0-harmless.json",
    ]
    
    # beta 2.0
    helpful_beta_2 = [
        "results/responses/sweep/win_rates/typo-iteration-0vsbase-lr-1e-6-beta-2.0-temperature-0.0-helpful.json",
        "results/responses/sweep/win_rates/typo-iteration-1vs0-lr-1e-6-beta-2.0-temperature-0.0-helpful.json",
        "results/responses/sweep/win_rates/typo-iteration-2vs1-lr-1e-6-beta-2.0-temperature-0.0-helpful.json",
    ]
    
    harmless_beta_2 = [
        "results/responses/sweep/win_rates/typo-iteration-0vsbase-lr-1e-6-beta-2.0-temperature-0.0-harmless.json",
        "results/responses/sweep/win_rates/typo-iteration-1vs0-lr-1e-6-beta-2.0-temperature-0.0-harmless.json",
        "results/responses/sweep/win_rates/typo-iteration-2vs1-lr-1e-6-beta-2.0-temperature-0.0-harmless.json",
    ]
    
    # beta 3.0
    helpful_beta_3 = [
        "results/responses/sweep/win_rates/typo-iteration-0vsbase-lr-1e-6-beta-3.0-temperature-0.0-helpful.json",
        "results/responses/sweep/win_rates/typo-iteration-1vs0-lr-1e-6-beta-3.0-temperature-0.0-helpful.json",
        "results/responses/sweep/win_rates/typo-iteration-1vs0-lr-1e-6-beta-3.0-temperature-0.0-helpful.json",
    ]
    
    harmless_beta_3 = [
        "results/responses/sweep/win_rates/typo-iteration-0vsbase-lr-1e-6-beta-3.0-temperature-0.0-harmless.json",
        "results/responses/sweep/win_rates/typo-iteration-1vs0-lr-1e-6-beta-3.0-temperature-0.0-harmless.json",
        "results/responses/sweep/win_rates/typo-iteration-1vs0-lr-1e-6-beta-3.0-temperature-0.0-harmless.json",
    ]

    # Load datasets
    dataset_helpful_beta_05 = [load_data(path) for path in helpful_beta_05]
    dataset_harmless_beta_05  = [load_data(path) for path in harmless_beta_05]
    dataset_helpful_beta_1 = [load_data(path) for path in helpful_beta_1]
    dataset_harmless_beta_1  = [load_data(path) for path in harmless_beta_1]
    dataset_helpful_beta_2 = [load_data(path) for path in helpful_beta_2]
    dataset_harmless_beta_2  = [load_data(path) for path in harmless_beta_2]
    dataset_helpful_beta_3 = [load_data(path) for path in helpful_beta_3]
    dataset_harmless_beta_3  = [load_data(path) for path in harmless_beta_3]

    # Calculate statistics
    means_helpful_beta_05, ns_helpful_beta_05, errors_helpful_beta_05 = calculate_statistics(dataset_helpful_beta_05)
    means_harmless_beta_05, ns_harmless_beta_05, errors_harmless_beta_05 = calculate_statistics(dataset_harmless_beta_05)
    means_helpful_beta_1, ns_helpful_beta_1, errors_helpful_beta_1 = calculate_statistics(dataset_helpful_beta_1)
    means_harmless_beta_1, ns_harmless_beta_1, errors_harmless_beta_1 = calculate_statistics(dataset_harmless_beta_1)
    means_helpful_beta_2, ns_helpful_beta_2, errors_helpful_beta_2 = calculate_statistics(dataset_helpful_beta_2)
    means_harmless_beta_2, ns_harmless_beta_2, errors_harmless_beta_2 = calculate_statistics(dataset_harmless_beta_2)
    means_helpful_beta_3, ns_helpful_beta_3, errors_helpful_beta_3 = calculate_statistics(dataset_helpful_beta_3)
    means_harmless_beta_3, ns_harmless_beta_3, errors_harmless_beta_3 = calculate_statistics(dataset_harmless_beta_3)
  
    
    # Plotting
    iterations = [1, 2, 3] 
    plot_results(iterations, [
        means_helpful_beta_05, means_helpful_beta_1, means_helpful_beta_2, means_helpful_beta_3],
                 [errors_helpful_beta_05, errors_helpful_beta_1, errors_helpful_beta_2, errors_helpful_beta_3],
                    ['beta 0.5', 'beta 1.0', 'beta 2.0', 'beta 3.0'], 'Helpful Win Rates', 'helpful_win_rates')
    
    plot_results(iterations, [
        means_harmless_beta_05, means_harmless_beta_1, means_harmless_beta_2, means_harmless_beta_3],
                 [errors_harmless_beta_05, errors_harmless_beta_1, errors_harmless_beta_2, errors_harmless_beta_3],
                    ['beta 0.5', 'beta 1.0', 'beta 2.0', 'beta 3.0'], 'Helpful Win Rates', 'harmless_win_rates')
    
    breakpoint()

if __name__ == "__main__":
    main()
