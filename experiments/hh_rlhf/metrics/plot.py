import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

BATCH_SIZE = 1
N_EXAMPLES = 20
COMMON = False
LOG_PROBS = True

def load_data(file_path):
    try:
        with open(file_path, "r") as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return []

# File paths
file_paths = [
    "predictions/rlhf_gen_mixtral_7b_instruct_eval_mistral_7b_base_run_2_model_mistral_7b_base_train.json",
    "predictions/rlhf_reversed_gen_mixtral_7b_instruct_eval_mistral_7b_base_run_2_model_mistral_7b_base_train.json",
    "predictions/rlhf_gen_mistral_7b_base_eval_mistral_7b_base_run_1_no_memory_model_mistral_7b_base.json",
    "predictions/rlhf_reversed_gen_mistral_7b_base_eval_mistral_7b_base_run_1_no_memory_model_mistral_7b_base.json",
    # "predictions/rlhf_gen_mistral_7b_instruct_eval_mistral_7b_base_run_2_model_mixtral_7b_instruct.json",
    # "predictions/rlhf_reversed_gen_mistral_7b_instruct_eval_mistral_7b_base_run_2_model_mixtral_7b_instruct.json",
]

# Loading data
datasets = [load_data(path) for path in file_paths]
breakpoint()

# Setting the theme and font
sns.set_theme(style="darkgrid")
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.figure(figsize=(10, 5))
colors = sns.palettes.color_palette("colorblind", 10)

# Calculate means and standard errors
def calculate_stats(values):
    if len(values) > 1:  # Ensure there are enough values for meaningful stats
        mean = np.median(values)
        error = 1.95 * (np.std(values) / np.sqrt(len(values)))
    else:
        mean = np.nan  # Use NaN to indicate insufficient data
        error = np.nan
    return mean, error


def calculate_win_rate_and_error(chosen, reversed_chosen):
    win_count = sum(1 for r, rev in zip(chosen, reversed_chosen) if r > rev)
    total = len(chosen)
    win_rate = win_count / total
    error = 1.96 * np.sqrt(win_rate * (1 - win_rate) / total)
    return win_rate, error


def find_common_indices(datasets):
    # Start with all indices as common
    common_indices = set(str(i) for i in range(N_EXAMPLES))

    # Intersect with indices available in each dataset
    for data in datasets:
        dataset_indices = set(data.keys())
        common_indices.intersection_update(dataset_indices)

    return common_indices
 # Fallback to default range if empty

# Identify common indices across all datasets

common_indices = find_common_indices(datasets)


if COMMON is True:
    chosen_means, chosen_errors = zip(*(calculate_stats([data[str(i)]["chosen"] for i in common_indices]) for data in datasets))
    rejected_means, rejected_errors = zip(*(calculate_stats([data[str(i)]["rejected"] for i in common_indices]) for data in datasets))
    n_evaluated_means = [np.mean([data[str(i)]["n_evaluated"] for i in common_indices])/BATCH_SIZE for data in datasets]
    actual_n_examples = len(common_indices)
    # Dataset names
    predictions = [
        f"rlhf-mixtral (N {actual_n_examples})", 
        f"rlhf-reversed-mixtral (N {actual_n_examples})", 
        f"rlhf-mistral (N {actual_n_examples})", 
        f"rlhf-reversed-mistral (N {actual_n_examples})", 
    ]
else:
    if LOG_PROBS:
       
        rlhf_chosen_mixtral = [datasets[0][str(i)]["chosen"] for i in datasets[0]]
        rlhf_rejected_mixtral = [datasets[0][str(i)]["rejected"] for i in datasets[0]]
        rlhf_reversed_chosen_mixtral = [datasets[1][str(i)]["chosen"] for i in datasets[1]]
        rlhf_reversed_rejected_mixtral = [datasets[1][str(i)]["rejected"] for i in datasets[1]]
        
        rlhf_chosen_mistral = [datasets[2][str(i)]["chosen"] for i in datasets[2]]
        rlhf_rejected_mistral = [datasets[2][str(i)]["rejected"] for i in datasets[2]]
        rlhf_reversed_chosen_mistral = [datasets[3][str(i)]["chosen"] for i in datasets[3]]
        rlhf_reversed_rejected_mistral = [datasets[3][str(i)]["rejected"] for i in datasets[3]]
        
        # Calculate win rates and errors for mixtral
        rlhf_mixtral_chosen_win_rate, rlhf_mixtral_chosen_error = calculate_win_rate_and_error(rlhf_chosen_mixtral, rlhf_reversed_chosen_mixtral)
        rlhf_mixtral_rejected_win_rate, rlhf_mixtral_rejected_error = calculate_win_rate_and_error(rlhf_rejected_mixtral, rlhf_reversed_rejected_mixtral)

        # Calculate win rates and errors for mistral (assuming rlhf_chosen_mistral and others are defined)
        rlhf_mistral_chosen_win_rate, rlhf_mistral_chosen_error = calculate_win_rate_and_error(rlhf_chosen_mistral, rlhf_reversed_chosen_mistral)
        rlhf_mistral_rejected_win_rate, rlhf_mistral_rejected_error = calculate_win_rate_and_error(rlhf_rejected_mistral, rlhf_reversed_rejected_mistral)

        # Combine win rates and errors
        win_rates_chosen = [rlhf_mixtral_chosen_win_rate, rlhf_mixtral_rejected_win_rate, rlhf_mistral_chosen_win_rate, rlhf_mistral_rejected_win_rate]
        win_rates_rejected = [1 - c for c in win_rates_chosen]
        win_errors = [rlhf_mixtral_chosen_error, rlhf_mixtral_rejected_error, rlhf_mistral_chosen_error, rlhf_mistral_rejected_error]

        # Names for x-axis labels
        predictions = [
            f"rlhf-mixtral (N {len(rlhf_chosen_mixtral)})", 
            f"rlhf-reversed-mixtral (N {len(rlhf_chosen_mixtral)})", 
            f"rlhf-mistral (N {len(rlhf_chosen_mistral)})", 
            f"rlhf-reversed-mistral (N {len(rlhf_chosen_mistral)})"]

        # Plotting
        fig, ax = plt.subplots(figsize=(10, 5))
        bar_width = 0.2
        opacity = 0.8
        colors = sns.palettes.color_palette("colorblind", 10)

        # Bar positions
        bar_pos = np.arange(len(predictions))
        bar_pos_chosen = [x - bar_width/2 for x in bar_pos]
        bar_pos_rejected = [x + bar_width/2 for x in bar_pos]

        # Bars for Chosen and Rejected
        ax.bar(bar_pos_chosen, win_rates_chosen, bar_width, alpha=opacity, color=colors[0], yerr=win_errors, label='% Chosen')
        ax.bar(bar_pos_rejected, win_rates_rejected, bar_width, alpha=opacity, color=colors[1], yerr=win_errors, label='% Rejected')

        # Labels, Title, and Custom x-axis
        ax.set_xlabel('Models')
        ax.set_ylabel('Win Rates')
        ax.set_title('RLHF Chosen vs Rejected Win Rates on Train Examples')
        ax.set_xticks(bar_pos)
        ax.set_xticklabels(predictions)
        ax.legend()

        plt.tight_layout()

        ax.set_ylim(-0.05, 1.05)
        plt.savefig('predictions_train.pdf')
        plt.savefig('predictions_train.png')
        


    # else:
    #     chosen_means, chosen_errors = zip(*(calculate_stats([data[str(i)]["chosen"] for i in data]) for data in datasets))
    #     rejected_means, rejected_errors = zip(*(calculate_stats([data[str(i)]["rejected"] for i in data]) for data in datasets))
    #     n_evaluated_means = [np.mean([data[str(i)]["n_evaluated"] for i in data])/BATCH_SIZE for data in datasets]
    
    
    # # Number of actual examples evaluated


    # # Plotting
    # fig, ax = plt.subplots(figsize=(10, 5))
    # bar_width = 0.2
    # opacity = 0.8

    # # Bar positions
    # bar_pos_chosen = np.arange(len(predictions))
    # bar_pos_rejected = [x + bar_width for x in bar_pos_chosen]
    # bar_pos_eval = [x + bar_width for x in bar_pos_rejected]
    
    # # Bars for Chosen
    # ax.bar(bar_pos_chosen, chosen_means, bar_width, alpha=opacity, color=colors[0], yerr=chosen_errors, label='% Chosen')

    # # Bars for Rejected
    # ax.bar(bar_pos_rejected, rejected_means, bar_width, alpha=opacity, color=colors[1], yerr=rejected_errors, label='% Rejected')

    # # Bars for N Evaluated
    # ax.bar(bar_pos_eval, n_evaluated_means, bar_width, alpha=opacity, color=colors[2], label='% Evaluated')

    # # Labels, Title and Custom x-axis
    # ax.set_xlabel('Constitutions (Batch Size 10)')
    # ax.set_ylabel('Percentage')
    # ax.set_title('Final Assistant Response on HH-RLHF (test split)')
    # ax.set_xticks([r + bar_width for r in range(len(predictions))])
    # ax.set_xticklabels(predictions)
    # ax.legend()

    # ax.set_ylim(-100.00, 1.0)

    # # Show plot
    # plt.tight_layout()
    # plt.savefig('predictions.pdf')
    # plt.savefig('predictions.png')
