import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json

# Assuming plot_utils.py defines these functions
from plot_utils import lighten_color, change_saturation, get_fancy_bbox

def load_data(path):
    return json.load(open(path))

def main():
    file_paths = [
        "results/win_rates_harmless-temperature-0.0-500-responses-train-constitutions.json",
        "results/win_rates_harmless-temperature-0.0-500-responses-test-constitutions.json",
        "results/win_rates_harmless-temperature-0.3-500-responses-train-constitutions.json",
        "results/win_rates_harmless-temperature-0.3-500-responses-test-constitutions.json",
        "results/win_rates_harmless-temperature-1.0-500-responses-train-constitutions.json",
        "results/win_rates_harmless-temperature-1.0-500-responses-test-constitutions.json"
    ]
    
    datasets = [load_data(path) for path in file_paths]
    
    means = [np.mean(dataset) for dataset in datasets]
    ns = [len(dataset) for dataset in datasets]
    errors = [np.sqrt(mean * (1 - mean) / n) for mean, n in zip(means, ns)]
    
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams["font.size"] = 24
    palette = sns.color_palette('colorblind')
    fig, ax = plt.subplots(figsize=(10, 5))
    bar_width = 0.11
    bar_positions = [0.2, 0.3, 0.7, 0.8, 1.2, 1.3]
    lighter_bar_color = lighten_color('black', 0.8)

    for i, _ in enumerate(means):
        color = palette[1] if i % 2 == 1 else palette[0]
    
        ax.bar(
            bar_positions[i], 
            means[i],
            bar_width, 
            color=change_saturation(color, 0.9),
            edgecolor='black', 
            linewidth=2, 
            zorder=99, 
            alpha=0.7,
        )
        
        ax.errorbar(
            bar_positions[i], 
            means[i],
            yerr=errors[i],
            fmt='none',  
            color=lighter_bar_color,
            capsize=1, 
            elinewidth=2, 
            zorder=100,
            ls='none',
        )

    
    for i, patch in enumerate(ax.patches):
        bb = patch.get_bbox()
        color = patch.get_facecolor()
        p_bbox = get_fancy_bbox(bb, "round,pad=-0.005,rounding_size=0.005", color, mutation_aspect=1.5, i=i)

        patch.remove()
        ax.add_patch(p_bbox)
      

    breakpoint()
    ax.set_ylabel('Win Rate (%)')
    ax.set_yticks([0.01, .25, .5, .75])
    ax.set_yticklabels(['0', '25', '50', '75'])
    ax.set_xlabel('Sampling Temperature')
    ax.set_xticks([0.25, .75, 1.25])
    ax.set_xticklabels(['0.0', '0.3', '1.0'])
    ax.set_ylim(0.01, .77)  
    # ax.legend(["Train", "Test"], loc='upper right', ncol=2,  bbox_to_anchor=(1, 1))
    sns.despine(left=True, top=True)
    ax.axhline(y=0.5, color='lightgrey', linestyle='--', linewidth=2, label='Chance')
    ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5, zorder=-100)

    plt.tight_layout()
    plt.savefig('results/harmless_win_rates.pdf')
    plt.savefig('results/harmless_win_rates.png')
    plt.show()

if __name__ == "__main__":
    main()
