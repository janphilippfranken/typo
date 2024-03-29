import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json

OUTPUT_DIR = 'results/plots'

def load_data(path):
    """Load data from a JSON file."""
    with open(path) as f:
        return json.load(f)

def calculate_statistics(datasets):
    """Calculate means, sample sizes, and standard errors for a list of datasets."""
    means = [np.mean(dataset) for dataset in datasets]
    ns = [len(dataset) for dataset in datasets]
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
    ax.set_xlabel('Iteration (t)')
    ax.set_xticks(temperatures)
    ax.set_xticklabels(map(str, temperatures))
    # ax.set_yticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    # ax.set_yticklabels([0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    # ax.set_ylim(0.375, .925)  
    ax.set_xticklabels(["Base", "1", "2", "3", "4"])
    ax.set_xlim(min(temperatures) - 0.3, max(temperatures) + 0.3)
    # ax.axhline(y=0.5, color='lightgrey', linestyle='--', linewidth=2)
    ax.legend(loc='upper left', ncol=2, fontsize=8, frameon=False, bbox_to_anchor=(0.25, 0.95), prop={'size': 16})

    sns.despine()
    plt.title(title)
    plt.tight_layout()
    
    plt.savefig(f'{OUTPUT_DIR}/{filename}_length_best.pdf')
    plt.savefig(f'{OUTPUT_DIR}/{filename}_length_best.png')
    plt.close()  # Close the plot to prevent it from showing inline if using Jupyter

def main():
    iterations = [0, 1, 2, 3, 4]
    labels = ['Helpful', 'Harmless']

    # Load datasets
    helpful_datasets = [
        load_data(path) for path in [
            "results/length/base-model-helpful.json",
            "results/length/1.0-helpful-iteration-0.json",
            "results/length/1.0-helpful-iteration-1.json",
            "results/length/1.0-helpful-iteration-2.json",
            "results/length/1.0-helpful-iteration-3.json",
        ]
    ]
    
    harmless_datasets = [
        load_data(path) for path in [
            "results/length/base-model-harmless.json",
            "results/length/1.0-harmless-iteration-0.json",
            "results/length/1.0-harmless-iteration-1.json",
            "results/length/1.0-harmless-iteration-2.json",
            "results/length/1.0-harmless-iteration-3.json",
        ]
    ]
    
    # Calculate statistics
    means_helpful, ns_helpful, errors_helpful = calculate_statistics(helpful_datasets)
    means_harmless, ns_harmless, errors_harmless = calculate_statistics(harmless_datasets)
    print(means_helpful)
    # print(errors_helpful)
    print(means_harmless)
    # print(errors_harmless)
    breakpoint()
    # Plotting
    plot_results(iterations, [means_helpful, means_harmless], [errors_helpful, errors_harmless], labels, "Length", "length_comparison")

if __name__ == "__main__":
    main()
