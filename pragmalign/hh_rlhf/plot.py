import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json

OUTPUT_DIR = 'results/responses'


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
    plt.rcParams["font.size"] = 28
    palette = sns.color_palette('colorblind')
    fig, ax = plt.subplots(figsize=(10, 10))

    for i, (means, errors) in enumerate(zip(means_ft, errors_ft)):
        ax.errorbar(temperatures, means, yerr=errors, fmt='o-', color=palette[i], label=labels[i])

    ax.set_ylabel('Win Rate')
    ax.set_xlabel('Sampling Temperature')
    ax.set_xticks(temperatures)
    ax.set_xticklabels(map(str, temperatures))
    ax.set_yticks([0.0, 0.25, 0.5,  0.75, 1.0])
    ax.set_yticklabels([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_ylim(-0.05, 1.0)  
    ax.set_xlim(min(temperatures) - 0.05, max(temperatures) + 0.05)
    ax.axhline(y=0.5, color='lightgrey', linestyle='--', linewidth=2)
    ax.legend(loc='upper left', ncol=2, fontsize=8, frameon=False, bbox_to_anchor=(0.01, 0.35), prop={'size': 12})

    sns.despine()
    plt.title(title)
    plt.tight_layout()
    
    plt.savefig(f'{OUTPUT_DIR}/{filename}.pdf')
    plt.savefig(f'{OUTPUT_DIR}/{filename}.png')
    plt.close()  # Close the plot to prevent it from showing inline if using Jupyter

def main():
    
    typo_base_helpful = [
        "results/win_rates_gpt4/stpo-beta-0.1-against-base-helpful-temperature-0.0.json",
        "results/win_rates_gpt4/stpo-beta-0.1-against-base-helpful-temperature-0.3.json",
        "results/win_rates_gpt4/stpo-beta-0.1-against-base-helpful-temperature-1.0.json",
    ]
    
    typo_base_harmless = [
        "results/win_rates_gpt4/stpo-beta-0.1-against-base-harmless-temperature-0.0.json",
        "results/win_rates_gpt4/stpo-beta-0.1-against-base-harmless-temperature-0.3.json",
        "results/win_rates_gpt4/stpo-beta-0.1-against-base-harmless-temperature-1.0.json",
    ]
    
    typo_dpo_helpful = [
        "results/win_rates_gpt4/stpo-beta-0.1-against-dpo-helpful-temperature-0.0.json",
        "results/win_rates_gpt4/stpo-beta-0.1-against-dpo-helpful-temperature-0.3.json",
        "results/win_rates_gpt4/stpo-beta-0.1-against-dpo-helpful-temperature-1.0.json",
    ]
    
    typo_dpo_harmless = [
        "results/win_rates_gpt4/stpo-beta-0.1-against-dpo-harmless-temperature-0.0.json",
        "results/win_rates_gpt4/stpo-beta-0.1-against-dpo-harmless-temperature-0.3.json",
        "results/win_rates_gpt4/stpo-beta-0.1-against-dpo-harmless-temperature-1.0.json",
    ]
    
    # typo_kto_helpful = [
    #     "results/win_rates_gpt4/stpo-beta-0.1-against-kto-helpful-temperature-0.0.json",
    #     "results/win_rates_gpt4/stpo-beta-0.1-against-kto-helpful-temperature-0.3.json",
    #     "results/win_rates_gpt4/stpo-beta-0.1-against-kto-helpful-temperature-0.3.json",
    # ]
    
    # typo_kto_harmless = [
    #     "results/win_rates_gpt4/stpo-beta-0.1-against-kto-harmless-temperature-0.0.json",
    #     "results/win_rates_gpt4/stpo-beta-0.1-against-kto-harmless-temperature-0.3.json",
    #     "results/win_rates_gpt4/stpo-beta-0.1-against-kto-harmless-temperature-0.3.json",
    # ]
    
    dpo_base_helpful = [
        "results/win_rates_gpt4/dpo-beta-0.1-against-base-helpful-temperature-0.0.json",
        "results/win_rates_gpt4/dpo-beta-0.1-against-base-helpful-temperature-0.3.json",
        "results/win_rates_gpt4/dpo-beta-0.1-against-base-helpful-temperature-1.0.json",
    ]
    
    dpo_base_harmless = [
        "results/win_rates_gpt4/dpo-beta-0.1-against-base-harmless-temperature-0.0.json",
        "results/win_rates_gpt4/dpo-beta-0.1-against-base-harmless-temperature-0.3.json",
        "results/win_rates_gpt4/dpo-beta-0.1-against-base-harmless-temperature-1.0.json",
    ]
    
    # kto_base_helpful = [
    #     "results/win_rates_gpt4/kto-beta-0.1-against-base-helpful-temperature-0.0.json",
    #     "results/win_rates_gpt4/kto-beta-0.1-against-base-helpful-temperature-0.3.json",
    #     "results/win_rates_gpt4/kto-beta-0.1-against-base-helpful-temperature-0.3.json",
    # ]
    
    # kto_base_harmless = [
    #     "results/win_rates_gpt4/kto-beta-0.1-against-base-harmless-temperature-0.0.json",
    #     "results/win_rates_gpt4/kto-beta-0.1-against-base-harmless-temperature-0.3.json",
    #     "results/win_rates_gpt4/kto-beta-0.1-against-base-harmless-temperature-0.3.json",
    # ]
    
  
    
    

    # Load datasets
    datasets_base_helpful = [load_data(path) for path in typo_base_helpful]
    datasets_base_harmless = [load_data(path) for path in typo_base_harmless]
    datasets_dpo_helpful = [load_data(path) for path in typo_dpo_helpful]
    datasets_dpo_harmless = [load_data(path) for path in typo_dpo_harmless]
    # datasets_kto_helpful = [load_data(path) for path in typo_kto_helpful]
    # datasets_kto_harmless = [load_data(path) for path in typo_kto_harmless]
    datasets_dpo_base_helpful = [load_data(path) for path in dpo_base_helpful]
    datasets_dpo_base_harmless = [load_data(path) for path in dpo_base_harmless]
    # datasets_kto_base_helpful = [load_data(path) for path in kto_base_helpful]
    # datasets_kto_base_harmless = [load_data(path) for path in kto_base_harmless]

    # Calculate statistics
    means_base_helpful, _, errors_base_helpful = calculate_statistics(datasets_base_helpful)
    means_base_harmless, _, errors_base_harmless = calculate_statistics(datasets_base_harmless)
    
    means_dpo_helpful, _, errors_dpo_helpful = calculate_statistics(datasets_dpo_helpful)
    means_dpo_harmless, _, errors_dpo_harmless = calculate_statistics(datasets_dpo_harmless)
    # means_kto_helpful, _, errors_kto_helpful = calculate_statistics(datasets_kto_helpful)
    # means_kto_harmless, _, errors_kto_harmless = calculate_statistics(datasets_kto_harmless)
    means_dpo_base_helpful, _, errors_dpo_base_helpful = calculate_statistics(datasets_dpo_base_helpful)
    means_dpo_base_harmless, _, errors_dpo_base_harmless = calculate_statistics(datasets_dpo_base_harmless)
    # means_kto_base_helpful, _, errors_kto_base_helpful = calculate_statistics(datasets_kto_base_helpful)
    # means_kto_base_harmless, _, errors_kto_base_harmless = calculate_statistics(datasets_kto_base_harmless)

    # Plotting
    temperatures = [0.0, 0.3, 1.0]  # Assuming these are your temperature values
    plot_results(temperatures, [
        means_base_helpful, means_dpo_helpful,  means_dpo_base_helpful], 
                 [errors_base_helpful, errors_dpo_helpful,  errors_dpo_base_helpful], 
                 ['typo > base', 'typo > dpo', 'dpo > base'], 'Helpful Win Rates', 'helpful_win_rates')
    
    plot_results(temperatures, [
        means_base_harmless, means_dpo_harmless, means_dpo_base_harmless], 
                 [errors_base_harmless, errors_dpo_harmless, errors_dpo_base_harmless], 
                 ['typo > base', 'typo > dpo', 'dpo > base'], 'Harmless Win Rates', 'harmless_win_rates')
    
    
    breakpoint()

if __name__ == "__main__":
    main()
