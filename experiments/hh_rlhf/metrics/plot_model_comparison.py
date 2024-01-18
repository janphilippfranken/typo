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

        answer = int(p_chosen_model - p_chosen_base > p_rejected_model - p_rejected_base)
        labels_model.append(answer)
    return labels_model


def main(args):
    all_labels_chosen = []
    all_labels_rejected = []
    for run in range(args.start, args.n_runs):
 
        file_paths = [
            # not trained
            f"model_comparison/rlhf_test_gen_mixtral_7b_base_eval_mixtral_7b_base_gen_prompt_generation_prompt_base_2_run_{run}_model_mixtral_7b_base_train_n_final_1_mcq_a_b.json",
            f"model_comparison/rlhf_reversed_test_gen_mixtral_7b_base_eval_mixtral_7b_base_gen_prompt_generation_prompt_base_2_run_{run}_model_mixtral_7b_base_train_n_final_1_mcq_a_b.json",
            
            f"model_comparison/rlhf_test_gen_mixtral_7b_base_eval_mixtral_7b_base_gen_prompt_generation_prompt_base_2_run_{run}_model_mixtral_7b_base_peft_cai_data_hh_rlhf_train_n_final_1_mcq_a_b.json",
            f"model_comparison/rlhf_reversed_test_gen_mixtral_7b_base_eval_mixtral_7b_base_gen_prompt_generation_prompt_base_2_run_{run}_model_mixtral_7b_base_peft_cai_data_hh_rlhf_train_n_final_1_mcq_a_b.json",
        
            f"model_comparison/rlhf_test_gen_mixtral_7b_base_eval_mixtral_7b_base_gen_prompt_generation_prompt_base_2_run_{run}_model_mixtral_7b_base_peft_cai_data_hh_rlhf_flipped_train_n_final_1_mcq_a_b.json",
            f"model_comparison/rlhf_reversed_test_gen_mixtral_7b_base_eval_mixtral_7b_base_gen_prompt_generation_prompt_base_2_run_{run}_model_mixtral_7b_base_peft_cai_data_hh_rlhf_flipped_train_n_final_1_mcq_a_b.json",
            
            f"model_comparison/rlhf_test_gen_mixtral_7b_base_eval_mixtral_7b_base_gen_prompt_generation_prompt_base_2_run_{run}_model_mixtral_7b_base_peft_cai_data_hh_rlhf_combined_train_n_final_1_mcq_a_b.json",
            f"model_comparison/rlhf_reversed_test_gen_mixtral_7b_base_eval_mixtral_7b_base_gen_prompt_generation_prompt_base_2_run_{run}_model_mixtral_7b_base_peft_cai_data_hh_rlhf_combined_train_n_final_1_mcq_a_b.json",      
        ]

        # Loading data
        datasets = [load_data(path) for path in file_paths]
        # Loading data

        # breakpoint()
        # Extract labels and calculate errors
        labels_raw = [dataset['0']['train_labels'] for dataset in datasets]
        # breakpoint()
        # breakpoint()
        # Extracting individual entries for base models
        base_model_probs_one = labels_raw[0]
        base_model_probs_two = labels_raw[1]

        # Extracting individual entries for hh model
        hh_model_probs_one = labels_raw[2]
        hh_model_probs_two = labels_raw[3]

        # Extracting individual entries for flipped model
        flipped_model_probs_one = labels_raw[4]
        flipped_model_probs_two = labels_raw[5]

        # Extracting individual entries for combined model
        combined_model_probs_one = labels_raw[6]
        combined_model_probs_two = labels_raw[7]  # Assuming there's an 8th entry

        # base_model_probs = [prob for sublist in labels_raw[:2] for prob in sublist]
        # hh_model_probs = [prob for sublist in labels_raw[2:4] for prob in sublist]
        # flipped_model_probs = [prob for sublist in labels_raw[4:6] for prob in sublist]
        # combined_model_probs = [prob for sublist in labels_raw[6:] for prob in sublist]

        # labels_hh_one = calculate_labels(base_model_probs_one, hh_model_probs_one)
        # labels_hh_two = calculate_labels(base_model_probs_two, hh_model_probs_two)

        # labels_flipped_one = calculate_labels(base_model_probs_one, flipped_model_probs_one)
        # labels_flipped_two = calculate_labels(base_model_probs_two, flipped_model_probs_two)

        # labels_combined_one = calculate_labels(base_model_probs_one, combined_model_probs_one)
        # labels_combined_two = calculate_labels(base_model_probs_two, combined_model_probs_two)

        # # Combined results
        # labels_corrected = [labels_hh_one, labels_hh_two, labels_flipped_one, labels_flipped_two, labels_combined_one, labels_combined_two]
        labels_corrected = labels_raw

        
        labels_chosen = [1 - np.mean(label) for label in labels_corrected]
        labels_rejected = [np.mean(label) for label in labels_corrected]
        all_labels_chosen.append(labels_chosen)
        all_labels_rejected.append(labels_rejected)
        
    breakpoint()
    
    labels = []
    errors = []
    # Assuming all_labels_chosen and all_labels_rejected are already populated lists
    all_labels_chosen = np.array(all_labels_chosen)
    all_labels_rejected = np.array(all_labels_rejected)  # Corrected this line

    # Calculate means
    all_labels_chosen_mean = list(all_labels_chosen.mean(axis=0))
    all_labels_rejected_mean = list(all_labels_rejected.mean(axis=0))

    # Calculate SEM (Standard Error of the Mean)
    all_labels_chosen_error = list(1.95 * all_labels_chosen.std(axis=0) / np.sqrt(all_labels_chosen.shape[0]))
    all_labels_rejected_error = list(1.95 * all_labels_rejected.std(axis=0) / np.sqrt(all_labels_rejected.shape[0]))
    print(all_labels_rejected.shape[0])
    
    for chosen, rejected, error_chosen, error_rejected in zip(all_labels_chosen_mean, all_labels_rejected_mean, all_labels_chosen_error, all_labels_rejected_error):
        labels.append(chosen)
        labels.append(rejected)
        errors.append(error_chosen)
        errors.append(error_rejected)
    
    
    # Names for x-axis labels
    predictions = [
        "mixtral-base",
        "mixtral-peft-hh-rlhf",
        "mixtral-peft-hh-rlhf-flipped",
        "mixtral-peft-hh-rlhf-combined",
    ]
    # Setting the theme and font
    sns.set_theme(style="darkgrid")
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.figure(figsize=(10, 5))
    colors = sns.palettes.color_palette("colorblind", 10)
    
    
    # Plotting
    fig, ax = plt.subplots(figsize=(15, 5))
    bar_width = 0.15
    opacity = 0.8
    color_palette = sns.color_palette("colorblind", 4)

                
            # Constants
    n_models = 4
    bars_per_model = 4  # Chosen | Constitution, Rejected | Constitution, Chosen | Flipped, Rejected | Flipped
    total_bars = n_models * bars_per_model

    bar_width = 0.15
    model_gap = 1.0
    constitution_gap = 0.2

    # Calculate positions
    positions = []
    current_pos = 0
    for _ in range(n_models):
        # For each model, add positions for its four bars
        positions.extend([current_pos, current_pos + bar_width, current_pos + bar_width + constitution_gap, current_pos + 2 * bar_width + constitution_gap])
        current_pos += 2 * bar_width + constitution_gap + model_gap

    # Plotting bars
    color = 0
    hatch_idx = 0
    for i in range(total_bars):
        model_index = i // bars_per_model
        if color == 0: 
            color = 1
        else: color = 0
        print(hatch_idx)
        hatch = '//' if hatch_idx < 2 else ''  # Chosen bars are hatched
        hatch_idx += 1
        if hatch_idx == 4:
            hatch_idx = 0

        ax.bar(positions[i], labels[i], bar_width, alpha=opacity, color=color_palette[color], hatch=hatch, yerr=errors[i])

    # Adjust the x-axis labels to be centered for each model
    model_centers = [(positions[i * bars_per_model] + positions[(i + 1) * bars_per_model - 1]) / 2 for i in range(n_models)]

    # Setting axis labels and title
    ax.set_xticks(model_centers)
    ax.set_xticklabels(predictions)
    ax.set_ylabel('Percentage (%)')
    ax.set_xlabel("Model")
    ax.set_title('p_chosen > p_rejected')
    ax.set_ylim(-0.05, 1.05)
    plt.tight_layout()
    ax.legend(["Chosen | Constitution", "Rejected | Constitution", "Chosen | Flipped Constitution", "Rejected | Flipped Constitution"])

    # Save plot
    plt.savefig(f'./model_comparison/test_results_a_b.pdf')
    plt.savefig(f'./model_comparison/test_results_a_b.png')

    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--n_runs', default=11, type=int, help='Number of runs to process')
    parser.add_argument('--start',  default=1, type=int, help='Start')
    parser.add_argument('--n_final',  default=1, type=int, help='Start')
    args = parser.parse_args()
    main(args)