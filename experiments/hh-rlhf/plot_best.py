import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json

OUTPUT_DIR = 'results/plots'


from plot_utils import lighten_color, change_saturation, get_fancy_bbox

def load_data(path):
    return json.load(open(path))

def calculate_statistics(datasets):
    """Calculate sample proportions and their 95% confidence interval errors for a list of binomial datasets."""
    z = 1.96  # z-score for 95% confidence
    means = []  # Sample proportions
    errors = []  # Confidence interval half-widths
    
    for dataset in datasets:
        n = len(dataset)  # Number of trials
        print(n)
        p_hat = np.mean(dataset)  # Sample proportion (success rate)
        se = np.sqrt(p_hat * (1 - p_hat) / n)  
        error = z * se  
        
        means.append(p_hat)
        errors.append(error)
    
    return means, None, errors

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
    ax.set_xticklabels(map(str,  temperatures))
    ax.set_yticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    ax.set_yticklabels([0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    ax.set_ylim(0.375, .925)  
    ax.set_xlim(.7, 4.3)
    ax.axhline(y=0.5, color='lightgrey', linestyle='--', linewidth=2)
    ax.legend(loc='upper left', ncol=2, fontsize=8, frameon=False, bbox_to_anchor=(0.25, 0.95), prop={'size': 16})

    sns.despine()
    plt.title(title)
    plt.tight_layout()
    
    plt.savefig(f'{OUTPUT_DIR}/{filename}_best.pdf')
    plt.savefig(f'{OUTPUT_DIR}/{filename}_best.png')
    plt.close()  # Close the plot to prevent it from showing inline if using Jupyter

def main(): 
    # beta 1.0
    helpful_beta_1 = [
        "results/responses/sweep/win_rates/typo-iteration-0vsbase-lr-1e-6-beta-1.0-temperature-0.0-helpful.json",
        "results/responses/sweep/win_rates/typo-iteration-1vs0-lr-1e-6-beta-1.0-temperature-0.0-helpful.json",
        "results/responses/sweep/win_rates/typo-iteration-2vs1-lr-1e-6-beta-1.0-temperature-0.0-helpful.json",
        "results/responses/sweep/win_rates/typo-iteration-3vs2-lr-1e-6-beta-1.0-temperature-0.0-helpful.json",
    ]
    
    harmless_beta_1 = [
        "results/responses/sweep/win_rates/typo-iteration-0vsbase-lr-1e-6-beta-1.0-temperature-0.0-harmless.json",
        "results/responses/sweep/win_rates/typo-iteration-1vs0-lr-1e-6-beta-1.0-temperature-0.0-harmless.json",
        "results/responses/sweep/win_rates/typo-iteration-2vs1-lr-1e-6-beta-1.0-temperature-0.0-harmless.json",
        "results/responses/sweep/win_rates/typo-iteration-3vs2-lr-1e-6-beta-1.0-temperature-0.0-harmless.json",
    ]
    
    # beta 1.0
    helpful_beta_1_base = [
        "results/responses/sweep/win_rates/typo-iteration-0vsbase-lr-1e-6-beta-1.0-temperature-0.0-helpful.json",
        "results/responses/sweep/win_rates/typo-iteration-1vsbase-lr-1e-6-beta-1.0-temperature-0.0-helpful.json",
        "results/responses/sweep/win_rates/typo-iteration-2vsbase-lr-1e-6-beta-1.0-temperature-0.0-helpful.json",
        "results/responses/sweep/win_rates/typo-iteration-3vsbase-lr-1e-6-beta-1.0-temperature-0.0-helpful.json",
    ]
    
    harmless_beta_1_base = [
        "results/responses/sweep/win_rates/typo-iteration-0vsbase-lr-1e-6-beta-1.0-temperature-0.0-harmless.json",
        "results/responses/sweep/win_rates/typo-iteration-1vsbase-lr-1e-6-beta-1.0-temperature-0.0-harmless.json",
        "results/responses/sweep/win_rates/typo-iteration-2vsbase-lr-1e-6-beta-1.0-temperature-0.0-harmless.json",
        "results/responses/sweep/win_rates/typo-iteration-3vsbase-lr-1e-6-beta-1.0-temperature-0.0-harmless.json",
    ]
    
  
    # Load datasets
 
    dataset_helpful_beta_1 = [load_data(path) for path in helpful_beta_1]
    dataset_harmless_beta_1  = [load_data(path) for path in harmless_beta_1]
    dataset_helpful_beta_1_base = [load_data(path) for path in helpful_beta_1_base]
    dataset_harmless_beta_1_base  = [load_data(path) for path in harmless_beta_1_base]
  

    # Calculate statistics
    means_helpful_beta_1, ns_helpful_beta_1, errors_helpful_beta_1 = calculate_statistics(dataset_helpful_beta_1)
    means_harmless_beta_1, ns_harmless_beta_1, errors_harmless_beta_1 = calculate_statistics(dataset_harmless_beta_1)
    means_helpful_beta_1_base, ns_helpful_beta_1_base, errors_helpful_beta_1_base = calculate_statistics(dataset_helpful_beta_1_base)
    means_harmless_beta_1_base, ns_harmless_beta_1_base, errors_harmless_beta_1_base= calculate_statistics(dataset_harmless_beta_1_base)
   
  
    
    # Plotting
    iterations = [1, 2, 3, 4] 
    plot_results(iterations, [
        means_helpful_beta_1_base, means_helpful_beta_1],
                 [errors_helpful_beta_1_base, errors_helpful_beta_1],
                    [r"$\text{typo}_{\text{t}}$ > base", r"$\text{typo}_{\text{t}}$ > $\text{typo}_{\text{t-1}}$"], 'Helpful Win Rates', 'helpful_win_rates')
    
    plot_results(iterations, [
        means_harmless_beta_1_base, means_harmless_beta_1],
                 [errors_harmless_beta_1_base, errors_harmless_beta_1],
                    [r"$\text{typo}_{\text{t}}$ > base", r"$\text{typo}_{\text{t}}$ > $\text{typo}_{\text{t-1}}$"], 'Harmless Win Rates', 'harmless_win_rates')
    
    breakpoint()

if __name__ == "__main__":
    main()
