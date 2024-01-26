import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

def load_data(file_path):
    try:
        with open(file_path, "r") as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return []
    
def calculate_error(log_probs):
    """
    Calculate the standard error for log probabilities.
    """
    variance = np.var(log_probs, ddof=1)  # ddof=1 for sample variance
    standard_error = np.sqrt(variance / len(log_probs))
    return standard_error


# Function to calculate labels
def calculate_labels(base_probs, model_probs):
    labels_model = []
    for base_prob, model_prob in zip(base_probs, model_probs):
        p_chosen_base = base_prob[0]
        p_rejected_base = base_prob[1]

        p_chosen_model = model_prob[0]
        p_rejected_model = model_prob[1]

        # answer = int(p_rejected_model - p_rejected_base > p_chosen_model - p_chosen_base)
        answer = int(p_chosen_model - p_chosen_base > p_rejected_model - p_rejected_base)
        labels_model.append(answer)
    return labels_model


def main(args):
    all_labels_chosen = []
    all_labels_rejected = []
    for run in range(args.start, args.n_runs):
 
        file_paths = [
            # not trained
            f"predictions/rlhf_test_mixtral_7b_base_run_{run}_model_mixtral_7b_base_test_no_constitution_answer.json",
            f"predictions/rlhf_test_mixtral_7b_dpo_16bit_run_{run}_model_mixtral_7b_dpo_16_bit_test_answer.json",
            f"predictions/rlhf_reversed_test_mixtral_7b_dpo_16bit_run_{run}_model_mixtral_7b_dpo_16_bit_test_answer.json",
        ]

        # Loading data
        datasets = [load_data(path) for path in file_paths]

    
        # # breakpoint()
        # breakpoint()
        labels_probs_base = datasets[0]['0']['train_logprobs']
        labels_probs_dpo_chosen = datasets[1]['0']['train_logprobs']
        labels_probs_dpo_rejected = datasets[2]['0']['train_logprobs']
        
        def compare(base, model):
        
            base_chosen = base[0]
            base_rejected = base[1]
            
            model_chosen = model[0]
            model_rejected = model[1]
            
            return int(model_chosen - base_chosen > model_rejected - base_rejected)
        
        breakpoint()
        
        labels = calculate_labels(labels_probs_base, labels_probs_dpo)
        # breakpoint()

        
        labels_chosen = np.mean(labels)
        labels_rejected = 1 - np.mean(labels)
        all_labels_chosen.append(labels_chosen)
        all_labels_rejected.append(labels_rejected)
        
    # breakpoint()
    
    breakpoint()
    labels = []
    errors = []
    # Assuming all_labels_chosen and all_labels_rejected are already populated lists
    all_labels_chosen = np.array(all_labels_chosen)
    all_labels_rejected = np.array(all_labels_rejected)  # Corrected this line

    # Calculate means
    all_labels_chosen_mean = np.mean(all_labels_chosen)
    all_labels_rejected_mean = np.mean(all_labels_rejected)

    # Calculate SEM (Standard Error of the Mean)
    # breakpoint()
    all_labels_chosen_error = 1.95 * all_labels_chosen.std() / np.sqrt(all_labels_chosen.shape[0])
    all_labels_rejected_error = 1.95 * all_labels_rejected.std() / np.sqrt(all_labels_rejected.shape[0])
    print(len(all_labels_chosen))

    
    # Setting the theme and font
    sns.set_theme(style="darkgrid")
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.figure(figsize=(10, 5))
    colors = sns.palettes.color_palette("colorblind", 10)
    
    # Plotting
    fig, ax = plt.subplots(figsize=(7, 5))
    bar_width = 0.24  # Adjusted bar width
    opacity = 0.8
    color_palette = sns.color_palette("colorblind", 4)

    print(all_labels_chosen_mean, all_labels_rejected_mean)
   
    ax.bar(0.25, all_labels_chosen_mean, bar_width, alpha=opacity, color=color_palette[0], yerr=all_labels_chosen_error)
    ax.bar(0.75, all_labels_rejected_mean, bar_width, alpha=opacity, color=color_palette[1], yerr=all_labels_rejected_error)

    # Setting axis labels and title
    ax.set_ylabel('Percentage (%)')
    ax.set_xticks([0.25, 0.75])
    ax.set_xticklabels(["", ""])
    # ax.set_xlabel("Chosen")
    ax.set_ylim(-0.05, 1.05)
    plt.tight_layout()
    ax.legend(["p_c_dpo - p_c_base > p_r_dpo - p_r_base", "p_c_dpo - p_c_base < p_r_dpo - p_r_base"], fontsize=12)
    # ax.legend(["p_r_dpo - p_r_base > p_c_dpo - p_c_base", "p_r_dpo - p_r_base < p_c_dpo - p_c_base"], fontsize=12)
    # Save plot
    plt.savefig(f'./predictions/test_reversed_nc.pdf')
    plt.savefig(f'./predictions/test_reversed_nc.png')

    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--n_runs', default=4, type=int, help='Number of runs to process')
    parser.add_argument('--start',  default=1, type=int, help='Start')
    args = parser.parse_args()
    main(args)