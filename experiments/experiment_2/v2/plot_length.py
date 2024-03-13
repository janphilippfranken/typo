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

def plot_results(categories, means_ft, errors_ft, labels, title, filename):
    """Plot bar charts for each feature type (ft) with error bars."""
    plt.rcParams["font.size"] = 12
    palette = sns.color_palette('colorblind')
    fig, ax = plt.subplots(figsize=(12, 6))

    bar_width = 0.35 / len(labels)  # Adjust bar width to fit the new number of bars
    index = np.arange(len(categories))
    
    for i, (means, errors) in enumerate(zip(means_ft, errors_ft)):
        positions = index + bar_width * i - bar_width / 2  # Adjust positions for two bars per category
        ax.bar(positions, means, bar_width, yerr=errors, color=palette[i], label=labels[i], capsize=5, edgecolor='grey')

    ax.set_ylabel('Average Response Tokens')
    ax.set_xlabel('Model')
    ax.set_xticks(index)
    ax.set_xticklabels(categories)
    # ax.set_ylim(0, 1)  # Adjust based on your data range
    ax.legend()

    # ax.axhline(y=0.5, color='lightgrey', linestyle='--', linewidth=2)
    sns.despine()
    plt.title(title)
    plt.tight_layout()
    
    plt.savefig(f'{OUTPUT_DIR}/{filename}_length.pdf')
    plt.savefig(f'{OUTPUT_DIR}/{filename}_length.png')
    plt.close()

def main():
    categories = ['SFT', 'DPO', 'DPO + SFT', 'TYPO']
    labels = ['Helpful', 'Harmless']

    # Load datasets
    helpful_datasets = [
        load_data(path) for path in [
            "results/length/sft-positive-helpful-length.json",
            "results/length/dpo-no-sft-beta-0.1-helpful-length.json",
            "results/length/dpo-sft-both-beta-0.1-helpful-length.json",
            "results/length/typo-beta-0.3-helpful-length.json",
        ]
    ]
    
    harmless_datasets = [
        load_data(path) for path in [
            "results/length/sft-positive-harmless-length.json",
            "results/length/dpo-no-sft-beta-0.1-harmless-length.json",
            "results/length/dpo-sft-both-beta-0.1-harmless-length.json",
            "results/length/typo-beta-0.3-harmless-length.json",
        ]
    ]
    
    # Calculate statistics
    means_helpful, ns_helpful, errors_helpful = calculate_statistics(helpful_datasets)
    means_harmless, ns_harmless, errors_harmless = calculate_statistics(harmless_datasets)

    # Plotting
    plot_results(categories, [means_helpful, means_harmless], [errors_helpful, errors_harmless], labels, "Length", "length_comparison")

if __name__ == "__main__":
    main()
