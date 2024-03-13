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
    """Calculate sample proportions and their 95% confidence interval errors for a list of binomial datasets."""
    z = 1.96  # z-score for 95% confidence
    means = []  # Sample proportions
    errors = []  # Confidence interval half-widths
    
    for dataset in datasets:
        n = len(dataset)  # Number of trials
        print(n)
        p_hat = np.mean(dataset)  # Sample proportion (success rate)
        se = np.sqrt(p_hat * (1 - p_hat) / n)  # Standard error
        error = z * se  # Confidence interval half-width
        
        means.append(p_hat)
        errors.append(error)
    
    return means, None, errors


def plot_results(categories, means_ft, errors_ft, labels, title, filename):
    """Plot bar charts for each feature type (ft) with error bars."""
    plt.rcParams["font.size"] = 12
    palette = sns.color_palette('colorblind')
    fig, ax = plt.subplots(figsize=(10, 6))

    bar_width = 0.35
    index = np.arange(len(categories))
    
    for i, (means, errors) in enumerate(zip(means_ft, errors_ft)):
        positions = index + bar_width * i
        ax.bar(positions, means, bar_width, yerr=errors, color=palette[i], label=labels[i], capsize=5)

    ax.set_ylabel('Win Rate')
    ax.set_xlabel('Comparison')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 1)  # Adjust based on your data range
    ax.legend()

    ax.axhline(y=0.5, color='lightgrey', linestyle='--', linewidth=2)
    sns.despine()
    plt.title(title)
    plt.tight_layout()
    
    plt.savefig(f'{OUTPUT_DIR}/{filename}_bars.pdf')
    plt.savefig(f'{OUTPUT_DIR}/{filename}_bars.png')
    plt.close()

def main():
    categories = ['Typo > SFT', 'Typo > DPO', 'Typo > DPO + SFT']
    labels = ['Helpful', 'Harmless']

    # Load datasets
    helpful_datasets = [
        load_data(path) for path in [
            "results/win_rates/typo-beta-0.3-vs-sft-positive-helpful.json",
            "results/win_rates/typo-beta-0.3-vs-dpo-no-sft-beta-0.3-positive-helpful.json",
            "results/win_rates/typo-beta-0.3-vs-dpo-sft-both-beta-0.5-helpful.json",
        ]
    ]
    
    harmless_datasets = [
        load_data(path) for path in [
            "results/win_rates/typo-beta-0.3-vs-sft-positive-harmless.json",
            "results/win_rates/typo-beta-0.3-vs-dpo-no-sft-beta-0.3-positive-harmless.json",
            "results/win_rates/typo-beta-0.3-vs-dpo-sft-both-beta-0.5-harmless.json",
        ]
    ]
    
    # Calculate statistics
    means_helpful, ns_helpful, errors_helpful = calculate_statistics(helpful_datasets)
    means_harmless, ns_harmless, errors_harmless = calculate_statistics(harmless_datasets)
    print(means_helpful)
    print(errors_helpful)
    print(means_harmless)
    print(errors_harmless)
    # Plotting
    plot_results(categories, [means_helpful, means_harmless], [errors_helpful, errors_harmless], labels, "Win Rate Comparison", "comparison")

if __name__ == "__main__":
    main()
