import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json

OUTPUT_DIR = 'results/v3'


from plot_utils import lighten_color, change_saturation, get_fancy_bbox

def load_data(path):
    return json.load(open(path))

def calculate_statistics(datasets):
    """Calculate means, sample sizes, and standard errors for a list of datasets."""
    means = [np.mean(dataset) for dataset in datasets]
    ns = [len(dataset) for dataset in datasets]
    errors = [1.96 * np.std(dataset, ddof=1) / np.sqrt(n) for dataset, n in zip(datasets, ns)]
    return means, ns, errors

def plot_results(temperatures, means_ft, errors_ft, labels, title, filename):
    """Plot the results with error bars for each feature type (ft) and save to PDF."""
    plt.rcParams["font.size"] = 24
    palette = sns.color_palette('colorblind')
    fig, ax = plt.subplots(figsize=(10, 5))

    for i, (means, errors) in enumerate(zip(means_ft, errors_ft)):
        ax.errorbar(temperatures, means, yerr=errors, fmt='o-', color=palette[i], label=labels[i])

    ax.set_ylabel('Win Rate')
    ax.set_xlabel('Sampling Temperature')
    ax.set_xticks(temperatures)
    ax.set_xticklabels(map(str, temperatures))
    ax.set_ylim(0.05, 0.95)  
    ax.set_xlim(min(temperatures) - 0.05, max(temperatures) + 0.05)
    ax.axhline(y=0.5, color='lightgrey', linestyle='--', linewidth=2)
    ax.legend(loc='upper left', ncol=2, fontsize=8, frameon=False, bbox_to_anchor=(0.01, 0.25), prop={'size': 12})

    sns.despine()
    plt.title(title)
    plt.tight_layout()
    
    plt.savefig(f'{OUTPUT_DIR}/{filename}.pdf')
    plt.savefig(f'{OUTPUT_DIR}/{filename}.png')
    plt.close()  # Close the plot to prevent it from showing inline if using Jupyter

def main():
    ft_1_helpful = [
        "results/v3/win-rates-helpful-temperature-0.0-500-responses-train-constitutions-0-shot-baseline-against-0-shot-ft-1-beta-0.1-epoch-1.0.json",
        "results/v3/win-rates-helpful-temperature-0.3-500-responses-train-constitutions-0-shot-baseline-against-0-shot-ft-1-beta-0.1-epoch-1.0.json",
        "results/v3/win-rates-helpful-temperature-1.0-500-responses-train-constitutions-0-shot-baseline-against-0-shot-ft-1-beta-0.1-epoch-1.0.json",
    ]
    
    ft_1_harmless = [
        "results/v3/win_rates_harmless-temperature-0.0-500-responses-train-constitutions-0-shot-baseline-against-0-shot-ft-1-beta-0.1-epoch-1.0.json",
        "results/v3/win_rates_harmless-temperature-0.3-500-responses-train-constitutions-0-shot-baseline-against-0-shot-ft-1-beta-0.1-epoch-1.0.json",
        "results/v3/win_rates_harmless-temperature-1.0-500-responses-train-constitutions-0-shot-baseline-against-0-shot-ft-1-beta-0.1-epoch-1.0.json",
    ]
    
    ft_2_helpful = [
        "results/v3/win-rates-helpful-temperature-0.0-500-responses-train-constitutions-0-shot-baseline-against-0-shot-ft-2-beta-0.1-epoch-1.0.json",
        "results/v3/win-rates-helpful-temperature-0.3-500-responses-train-constitutions-0-shot-baseline-against-0-shot-ft-2-beta-0.1-epoch-1.0.json",
        "results/v3/win-rates-helpful-temperature-1.0-500-responses-train-constitutions-0-shot-baseline-against-0-shot-ft-2-beta-0.1-epoch-1.0.json",
    ]
    
    ft_2_harmless = [
        "results/v3/win_rates_harmless-temperature-0.0-500-responses-train-constitutions-0-shot-baseline-against-0-shot-ft-2-beta-0.1-epoch-1.0.json",
        "results/v3/win_rates_harmless-temperature-0.3-500-responses-train-constitutions-0-shot-baseline-against-0-shot-ft-2-beta-0.1-epoch-1.0.json",
        "results/v3/win_rates_harmless-temperature-1.0-500-responses-train-constitutions-0-shot-baseline-against-0-shot-ft-2-beta-0.1-epoch-1.0.json",
    ]
    

    ft_1_helpful = [load_data(path) for path in ft_1_helpful]
    ft_1_harmless = [load_data(path) for path in ft_1_harmless]
    ft_2_helpful = [load_data(path) for path in ft_2_helpful]
    ft_2_harmless = [load_data(path) for path in ft_2_harmless]

    # Calculate statistics
    means_ft_1_helpful, _, errors_ft_1_helpful = calculate_statistics(ft_1_helpful)
    means_ft_1_harmless, _, errors_ft_1_harmless = calculate_statistics(ft_1_harmless)
    means_ft_2_helpful, _, errors_ft_2_helpful = calculate_statistics(ft_2_helpful)
    means_ft_2_harmless, _, errors_ft_2_harmless = calculate_statistics(ft_2_harmless)

    # Plotting
    temperatures = [0.0, 0.3, 1.0]  # Assuming these are your temperature values
    plot_results(temperatures, [means_ft_1_helpful, means_ft_2_helpful], [errors_ft_1_helpful, errors_ft_2_helpful], ['pragmalign-1', 'pragmalign-2'], 'Helpful Win Rates', 'helpful_win_rates')
    plot_results(temperatures, [means_ft_1_harmless, means_ft_2_harmless], [errors_ft_1_harmless, errors_ft_2_harmless], ['pragmalign-1', 'pragmalign-2'], 'Harmless Win Rates', 'harmless_win_rates')


if __name__ == "__main__":
    main()
