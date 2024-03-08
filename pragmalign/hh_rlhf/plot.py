import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json

OUTPUT_DIR = 'results'


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
    ax.set_xlabel('Sampling Temperature')
    ax.set_xticks(temperatures)
    ax.set_xticklabels(map(str, temperatures))
    ax.set_yticks([0.5,  0.6, 0.7, 0.8, 0.9])
    ax.set_yticklabels([0.5, 0.6, 0.7, 0.8, 0.9])
    ax.set_ylim(0.475, .925)  
    ax.set_xlim(min(temperatures) - 0.05, max(temperatures) + 0.05)
    ax.axhline(y=0.5, color='lightgrey', linestyle='--', linewidth=2)
    ax.legend(loc='upper left', ncol=2, fontsize=8, frameon=False, bbox_to_anchor=(0.01, 0.35), prop={ 'size': 12})

    sns.despine()
    plt.title(title)
    plt.tight_layout()
    
    plt.savefig(f'{OUTPUT_DIR}/{filename}.pdf')
    plt.savefig(f'{OUTPUT_DIR}/{filename}.png')
    plt.close()  # Close the plot to prevent it from showing inline if using Jupyter

def main():
    # against base
    typo_base_helpful = [
        "results/win_rates_gpt4_no_sorry_positive/typo-beta-0.1-against-base-helpful-temperature-0.0.json",
        "results/win_rates_gpt4_no_sorry_positive/typo-beta-0.1-against-base-helpful-temperature-0.3.json",
        "results/win_rates_gpt4_no_sorry_positive/typo-beta-0.1-against-base-helpful-temperature-1.0.json",
    ]
    
    typo_base_harmless = [
        "results/win_rates_gpt4_no_sorry_positive/typo-beta-0.1-against-base-harmless-temperature-0.0.json",
        "results/win_rates_gpt4_no_sorry_positive/typo-beta-0.1-against-base-harmless-temperature-0.3.json",
        "results/win_rates_gpt4_no_sorry_positive/typo-beta-0.1-against-base-harmless-temperature-1.0.json",
    ]
    
    sft_base_helpful = [
        "results/win_rates_gpt4_no_sorry_positive/sft-against-base-helpful-temperature-0.0.json",
        "results/win_rates_gpt4_no_sorry_positive/sft-against-base-helpful-temperature-0.3.json",
        "results/win_rates_gpt4_no_sorry_positive/sft-against-base-helpful-temperature-1.0.json",
    ]
    
    sft_base_harmless = [
        "results/win_rates_gpt4_no_sorry_positive/sft-against-base-harmless-temperature-0.0.json",
        "results/win_rates_gpt4_no_sorry_positive/sft-against-base-harmless-temperature-0.3.json",
        "results/win_rates_gpt4_no_sorry_positive/sft-against-base-harmless-temperature-1.0.json",
    ]
    
    dpo_base_helpful = [
        "results/win_rates_gpt4_no_sorry_positive/sft-dpo-beta-0.1-against-base-helpful-temperature-0.0.json",
        "results/win_rates_gpt4_no_sorry_positive/sft-dpo-beta-0.1-against-base-helpful-temperature-0.3.json",
        "results/win_rates_gpt4_no_sorry_positive/sft-dpo-beta-0.1-against-base-helpful-temperature-1.0.json",
    ]
    
    dpo_base_harmless = [
        "results/win_rates_gpt4_no_sorry_positive/sft-dpo-beta-0.1-against-base-harmless-temperature-0.0.json",
        "results/win_rates_gpt4_no_sorry_positive/sft-dpo-beta-0.1-against-base-harmless-temperature-0.3.json",
        "results/win_rates_gpt4_no_sorry_positive/sft-dpo-beta-0.1-against-base-harmless-temperature-1.0.json",
    ]
    
    # dpo_sft_base_helpful = [
    #     "results/win_rates_gpt4_no_sorry_positive/dpo-beta-0.1-against-base-helpful-temperature-0.0.json",
    #     "results/win_rates_gpt4_no_sorry_positive/dpo-beta-0.1-against-base-helpful-temperature-0.3.json",
    #     "results/win_rates_gpt4_no_sorry_positive/dpo-beta-0.1-against-base-helpful-temperature-1.0.json",
    # ]
    
    # dpo_sft_base_harmless = [
    #     "results/win_rates_gpt4_no_sorry_positive/dpo-beta-0.1-against-sft-harmless-temperature-0.0.json",
    #     "results/win_rates_gpt4_no_sorry_positive/dpo-beta-0.1-against-sft-harmless-temperature-0.3.json",
    #     "results/win_rates_gpt4_no_sorry_positive/dpo-beta-0.1-against-sft-harmless-temperature-1.0.json",
    # ]
    
    # against others
    typo_sft_base_helpful = [
        "results/win_rates_gpt4_no_sorry_positive/typo-beta-0.1-against-sft-helpful-temperature-0.0.json",
        "results/win_rates_gpt4_no_sorry_positive/typo-beta-0.1-against-sft-helpful-temperature-0.3.json",
        "results/win_rates_gpt4_no_sorry_positive/typo-beta-0.1-against-sft-helpful-temperature-1.0.json",
    ]
    
    typo_sft_base_harmless = [
        "results/win_rates_gpt4_no_sorry_positive/typo-beta-0.1-against-sft-harmless-temperature-0.0.json",
        "results/win_rates_gpt4_no_sorry_positive/typo-beta-0.1-against-sft-harmless-temperature-0.3.json",
        "results/win_rates_gpt4_no_sorry_positive/typo-beta-0.1-against-sft-harmless-temperature-1.0.json",
    ]
    
    typo_dpo_base_helpful = [
        "results/win_rates_gpt4_no_sorry_positive/typo-beta-0.1-against-sft-dpo-helpful-temperature-0.0.json",
        "results/win_rates_gpt4_no_sorry_positive/typo-beta-0.1-against-sft-dpo-helpful-temperature-0.3.json",
        "results/win_rates_gpt4_no_sorry_positive/typo-beta-0.1-against-sft-dpo-helpful-temperature-1.0.json",
    ]
    
    typo_dpo_base_harmless = [
        "results/win_rates_gpt4_no_sorry_positive/typo-beta-0.1-against-sft-dpo-harmless-temperature-0.0.json",
        "results/win_rates_gpt4_no_sorry_positive/typo-beta-0.1-against-sft-dpo-harmless-temperature-0.3.json",
        "results/win_rates_gpt4_no_sorry_positive/typo-beta-0.1-against-sft-dpo-harmless-temperature-1.0.json",
    ]
    
 
  
    
    

    # Load datasets
    dataset_typo_base_helpful = [load_data(path) for path in typo_base_helpful]
    dataset_typo_base_harmless = [load_data(path) for path in typo_base_harmless]
    dataset_sft_base_helpful = [load_data(path) for path in sft_base_helpful]
    dataset_sft_base_harmless = [load_data(path) for path in sft_base_harmless]
    dataset_dpo_base_helpful = [load_data(path) for path in dpo_base_helpful]
    dataset_dpo_base_harmless = [load_data(path) for path in dpo_base_harmless]
    # dataset_dpo_sft_helpful = [load_data(path) for path in dpo_sft_base_helpful]
    # dataset_dpo_sft_harmless = [load_data(path) for path in dpo_sft_base_harmless]
    dataset_typo_sft_base_helpful = [load_data(path) for path in typo_sft_base_helpful]
    dataset_typo_sft_base_harmless = [load_data(path) for path in typo_sft_base_harmless]
    dataset_typo_dpo_base_helpful = [load_data(path) for path in typo_dpo_base_helpful]
    dataset_typo_dpo_base_harmless = [load_data(path) for path in typo_dpo_base_harmless]
  
  

    # Calculate statistics
    means_typo_base_helpful, ns_typo_base_helpful, errors_typo_base_helpful = calculate_statistics(dataset_typo_base_helpful)
    means_typo_base_harmless, ns_typo_base_harmless, errors_typo_base_harmless = calculate_statistics(dataset_typo_base_harmless)
    means_sft_base_helpful, ns_sft_base_helpful, errors_sft_base_helpful = calculate_statistics(dataset_sft_base_helpful)
    means_sft_base_harmless, ns_sft_base_harmless, errors_sft_base_harmless = calculate_statistics(dataset_sft_base_harmless)
    means_dpo_base_helpful, ns_dpo_base_helpful, errors_dpo_base_helpful = calculate_statistics(dataset_dpo_base_helpful)
    means_dpo_base_harmless, ns_dpo_base_harmless, errors_dpo_base_harmless = calculate_statistics(dataset_dpo_base_harmless)
    # means_dpo_sft_helpful, ns_dpo_sft_helpful, errors_dpo_sft_helpful = calculate_statistics(dataset_dpo_sft_helpful)
    # means_dpo_sft_harmless, ns_dpo_sft_harmless, errors_dpo_sft_harmless = calculate_statistics(dataset_dpo_sft_harmless)
    means_typo_sft_base_helpful, ns_typo_sft_base_helpful, errors_typo_sft_base_helpful = calculate_statistics(dataset_typo_sft_base_helpful)
    means_typo_sft_base_harmless, ns_typo_sft_base_harmless, errors_typo_sft_base_harmless = calculate_statistics(dataset_typo_sft_base_harmless)
    means_typo_dpo_base_helpful, ns_typo_dpo_base_helpful, errors_typo_dpo_base_helpful = calculate_statistics(dataset_typo_dpo_base_helpful)
    means_typo_dpo_base_harmless, ns_typo_dpo_base_harmless, errors_typo_dpo_base_harmless = calculate_statistics(dataset_typo_dpo_base_harmless)
    

    
    # Plotting
    temperatures = [0.0, 0.3, 1.0]  # Assuming these are your temperature values
    plot_results(temperatures, [
        means_typo_base_helpful, means_dpo_base_helpful,  means_typo_dpo_base_helpful],
                 [errors_typo_base_helpful, errors_dpo_base_helpful,  errors_typo_dpo_base_helpful],
                    ['typo > base','dpo > base','typo > dpo'], 'Helpful Win Rates', 'helpful_win_rates')
    
    plot_results(temperatures, [
        means_typo_base_harmless, means_dpo_base_harmless, means_typo_dpo_base_harmless],
                 [errors_typo_base_harmless,  errors_dpo_base_harmless, errors_typo_dpo_base_harmless],
                    ['typo > base','dpo > base', 'typo > dpo'], 'Harmless Win Rates', 'harmless_win_rates')
    
    
    breakpoint()

if __name__ == "__main__":
    main()
