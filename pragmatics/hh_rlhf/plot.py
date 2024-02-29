import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json

# Assuming plot_utils.py defines these functions
from plot_utils import lighten_color, change_saturation, get_fancy_bbox

def load_data(path):
    return json.load(open(path))

def main():
    ft_0 = [
        "results/v2/win-rates-helpful-temperature-0.0-500-responses-train-constitutions-0-shot-baseline-against-0-shot-ft.json",
        "results/v2/win-rates-helpful-temperature-0.3-500-responses-train-constitutions-0-shot-baseline-against-0-shot-ft.json",
        "results/v2/win-rates-helpful-temperature-1.0-500-responses-train-constitutions-0-shot-baseline-against-0-shot-ft.json",
    ]
    
    ft_1 = [
        "results/v2/win-rates-helpful-temperature-0.0-500-responses-train-constitutions-0-shot-baseline-against-0-shot-ft-beta-1.0-epoch-0.84.json",
        "results/v2/win-rates-helpful-temperature-0.3-500-responses-train-constitutions-0-shot-baseline-against-0-shot-ft-beta-1.0-epoch-0.84.json",
        "results/v2/win-rates-helpful-temperature-1.0-500-responses-train-constitutions-0-shot-baseline-against-0-shot-ft-beta-1.0-epoch-0.84.json",
    ]
    
    
    base_1 = [
        "results/v2/win-rates-helpful-temperature-0.0-500-responses-train-constitutions-0-shot-baseline-against-1-shot-baseline.json",
        "results/v2/win-rates-helpful-temperature-0.3-500-responses-train-constitutions-0-shot-baseline-against-1-shot-baseline.json",
        "results/v2/win-rates-helpful-temperature-1.0-500-responses-train-constitutions-0-shot-baseline-against-1-shot-baseline.json",
    ]
    
   
    ft_0 = [load_data(path) for path in ft_0]
    base_1 = [load_data(path) for path in base_1]
    ft_1 = [load_data(path) for path in ft_1]
    
    means_ft_0 = [np.mean(dataset) for dataset in ft_0]
    means_base_1 = [np.mean(dataset) for dataset in base_1]
    means_ft_1 = [np.mean(dataset) for dataset in ft_1]
    
    ns_ft_0 = [len(dataset) for dataset in ft_0]
    ns_base_1 = [len(dataset) for dataset in base_1]
    ns_ft_1 = [len(dataset) for dataset in ft_1]
    
    error_ft_0 = [1.96 * np.std(dataset) / np.sqrt(n) for dataset, n in zip(ft_0, ns_ft_0)]
    error_base_1 = [1.96 * np.std(dataset) / np.sqrt(n) for dataset, n in zip(base_1, ns_base_1)]
    error_ft_1 = [1.96 * np.std(dataset) / np.sqrt(n) for dataset, n in zip(ft_1, ns_ft_1)]
    
    # plt.rcParams['font.family'] = 'Avenir'
    plt.rcParams["font.size"] = 24
    palette = sns.color_palette('colorblind')
    fig, ax = plt.subplots(figsize=(10, 5))
    
    
    # add line plot for each means and error 
    ax.errorbar([0.0 ,0.3, 1.0], means_ft_0, yerr=error_ft_0, fmt='o-', color=palette[0], label='Pragmatic-v1-0-shot')
    ax.errorbar([0.0 ,0.3, 1.0], means_base_1, yerr=error_base_1, fmt='o-', color=palette[1], label='Base-1-shot')
    ax.errorbar([0.0 ,0.3, 1.0], means_ft_1, yerr=error_ft_1, fmt='o-', color=palette[2], label='Pragmatic-v2-1-shot')


    breakpoint()
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

    ax.legend(loc='upper left', ncol=3, fontsize=8, frameon=False, bbox_to_anchor=(0.01, 0.25), prop={'size': 12})
    plt.tight_layout()
    
    plt.savefig('results/v2/helpful_win-rates-2.pdf')
    plt.savefig('results/v2/helpful_win-rates-2.png')
    plt.show()

if __name__ == "__main__":
    main()
