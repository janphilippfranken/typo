import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json

# Assuming plot_utils.py defines these functions
from plot_utils import lighten_color, change_saturation, get_fancy_bbox

def load_data(path):
    return json.load(open(path))

def main():
    ft_0_helpful = [
        "results/v3/win-rates-helpful-temperature-0.0-500-responses-train-constitutions-0-shot-baseline-against-0-shot-ft-1-beta-0.1-epoch-1.0.json",
        "results/v3/win-rates-helpful-temperature-0.3-500-responses-train-constitutions-0-shot-baseline-against-0-shot-ft-1-beta-0.1-epoch-1.0.json",
        "results/v3/win-rates-helpful-temperature-1.0-500-responses-train-constitutions-0-shot-baseline-against-0-shot-ft-1-beta-0.1-epoch-1.0.json",
    ]
    
    ft_0_harmless = [
        "results/v3/win_rates_harmless-temperature-0.0-500-responses-train-constitutions-0-shot-baseline-against-0-shot-ft-1-beta-0.1-epoch-1.0.json",
        "results/v3/win_rates_harmless-temperature-0.3-500-responses-train-constitutions-0-shot-baseline-against-0-shot-ft-1-beta-0.1-epoch-1.0.json",
        "results/v3/win_rates_harmless-temperature-1.0-500-responses-train-constitutions-0-shot-baseline-against-0-shot-ft-1-beta-0.1-epoch-1.0.json",
    ]
    
    # ft_1 = [
    #     "results/v3/win-rates-helpful-temperature-0.0-500-responses-train-constitutions-0-shot-baseline-against-0-shot-ft-beta-1.0-epoch-0.84.json",
    #     "results/v3/win-rates-helpful-temperature-0.3-500-responses-train-constitutions-0-shot-baseline-against-0-shot-ft-beta-1.0-epoch-0.84.json",
    #     "results/v3/win-rates-helpful-temperature-1.0-500-responses-train-constitutions-0-shot-baseline-against-0-shot-ft-beta-1.0-epoch-0.84.json",
    # ]
    
    
    # base_1 = [
    #     "results/v3/win-rates-helpful-temperature-0.0-500-responses-train-constitutions-0-shot-baseline-against-1-shot-baseline.json",
    #     "results/v3/win-rates-helpful-temperature-0.3-500-responses-train-constitutions-0-shot-baseline-against-1-shot-baseline.json",
    #     "results/v3/win-rates-helpful-temperature-1.0-500-responses-train-constitutions-0-shot-baseline-against-1-shot-baseline.json",
    # ]
    
   
    ft_0_helpful = [load_data(path) for path in ft_0_helpful]
    ft_0_harmless = [load_data(path) for path in ft_0_harmless]
    
    
    means_ft_0_helpful = [np.mean(dataset) for dataset in ft_0_helpful]
    means_ft_0_harmless = [np.mean(dataset) for dataset in ft_0_harmless]
   
    
    ns_ft_0_helpful = [len(dataset) for dataset in ft_0_helpful]
    ns_ft_0_harmless = [len(dataset) for dataset in ft_0_harmless]
    
    error_ft_0_helpful = [1.96 * np.std(dataset) / np.sqrt(n) for dataset, n in zip(ft_0_helpful, ns_ft_0_helpful)]
    error_ft_0_harmless = [1.96 * np.std(dataset) / np.sqrt(n) for dataset, n in zip(ft_0_harmless, ns_ft_0_harmless)]
    
    # plt.rcParams['font.family'] = 'Avenir'
    plt.rcParams["font.size"] = 24
    palette = sns.color_palette('colorblind')
    fig, ax = plt.subplots(figsize=(10, 5))
    
    
    # add line plot for each means and error 
    ax.errorbar([0.0 ,0.3, 1.0], means_ft_0_helpful, yerr=error_ft_0_helpful, fmt='o-', color=palette[0], label='Helpful Queries')
    ax.errorbar([0.0 ,0.3, 1.0], means_ft_0_harmless, yerr=error_ft_0_harmless, fmt='o-', color=palette[1], label='Harmless Queries')

    print(ns_ft_0_helpful, ns_ft_0_harmless)
    print()
    ax.set_ylabel('Win Rate')
    ax.set_yticks([0.1, 0.3, 0.5, 0.7, 0.9])
    ax.set_yticklabels(['0.1', '0.3', '0.5', '0.7', '0.9'])
    ax.set_xlabel('Sampling Temperature')
    ax.set_xticks([0.0, 0.3, 1.0])
    ax.set_xticklabels(['0.0', '0.3', '1.0'])
    ax.set_ylim(0.05, 0.95)  
    ax.set_xlim(-.05, 1.05)  
   
    sns.despine(top=True)
    ax.axhline(y=0.5, color='lightgrey', linestyle='--', linewidth=2)
    # ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5, zorder=-100)

    ax.legend(loc='upper left', ncol=2, fontsize=8, frameon=False, bbox_to_anchor=(0.01, 0.25), prop={'size': 12})
    plt.tight_layout()
    
    plt.savefig('results/v3/win-rates-t1.pdf')
    plt.savefig('results/v3/win-rates-t1.png')
    plt.show()

if __name__ == "__main__":
    main()
