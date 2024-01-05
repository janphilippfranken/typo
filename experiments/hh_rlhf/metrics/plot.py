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
    "predictions/rlhf_gen_mixtral_7b_instruct_eval_mistral_7b_base_run_4_model_mistral_7b_base_train.json",
    "predictions/rlhf_reversed_gen_mixtral_7b_instruct_eval_mistral_7b_base_run_4_model_mistral_7b_base_train.json",
    "predictions/rlhf_gen_mixtral_7b_instruct_eval_mistral_7b_base_run_4_model_mistral_7b_base_test.json",
    "predictions/rlhf_reversed_gen_mixtral_7b_instruct_eval_mistral_7b_base_run_4_model_mistral_7b_base_test.json",
    # "predictions/rlhf_gen_mistral_7b_instruct_eval_mistral_7b_base_run_2_model_mixtral_7b_instruct.json",
    # "predictions/rlhf_reversed_gen_mistral_7b_instruct_eval_mistral_7b_base_run_2_model_mixtral_7b_instruct.json",
]

# Loading data
datasets = [load_data(path) for path in file_paths]


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


def calculate_win_rate_and_error(chosen_minus_rejected_rlhf, chosen_minus_rejected_rlhf_reversed):
    win_count = sum(1 for r, rev in zip(chosen_minus_rejected_rlhf, chosen_minus_rejected_rlhf_reversed) if r > rev)
    total = len(chosen_minus_rejected_rlhf)
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


# Identify common indices across all datasets
common_indices = find_common_indices(datasets)


if LOG_PROBS:
    rlhf_chosen_mixtral_train = np.array([datasets[0][str(i)]["chosen"] for i in datasets[0]])
    rlhf_rejected_mixtral_train = np.array([datasets[0][str(i)]["rejected"] for i in datasets[0]])
    rlhf_reversed_chosen_mixtral_train = np.array([datasets[1][str(i)]["chosen"] for i in datasets[1]])
    rlhf_reversed_rejected_mixtral_train = np.array([datasets[1][str(i)]["rejected"] for i in datasets[1]])
    
    rlhf_chosen_mixtral_test = np.array([datasets[2][str(i)]["chosen"] for i in datasets[2]])
    rlhf_rejected_mixtral_test = np.array([datasets[2][str(i)]["rejected"] for i in datasets[2]])
    rlhf_reversed_chosen_mixtral_test = np.array([datasets[3][str(i)]["chosen"] for i in datasets[3]])
    rlhf_reversed_rejected_mixtral_test = np.array([datasets[3][str(i)]["rejected"] for i in datasets[3]])

    # Calculate win rates and errors for mixtral
    win_rate_train, error_train = calculate_win_rate_and_error(
        rlhf_chosen_mixtral_train - rlhf_rejected_mixtral_train, 
        rlhf_reversed_chosen_mixtral_train - rlhf_reversed_rejected_mixtral_train,
    )
    
    win_rate_test, error_test = calculate_win_rate_and_error(
        rlhf_chosen_mixtral_test - rlhf_rejected_mixtral_test, 
        rlhf_reversed_chosen_mixtral_test - rlhf_reversed_rejected_mixtral_test,
    )

    # Combine win rates and errors
    win_rates_chosen = [win_rate_train, win_rate_test]
    win_rates_rejected = [1 - win_rate_train, 1 - win_rate_test]
    win_errors = [error_train , error_test]

    # Names for x-axis labels
    predictions = [
        f"train split (N {len(rlhf_chosen_mixtral_train)})", 
        f"test split (N {len(rlhf_chosen_mixtral_test)})", 
    ]
    breakpoint()
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
    ax.bar(bar_pos_chosen, win_rates_chosen, bar_width, alpha=opacity, color=colors[0], yerr=win_errors, label=r'$p(\text{chosen}|\text{rlhf}) - p(\text{rejected}|\text{rlhf}) > $' '\n' r'$p(\text{chosen}|\text{rlhf\_reversed}) - p(\text{rejected}|\text{rlhf\_reversed})$')

    ax.bar(bar_pos_rejected, win_rates_rejected, bar_width, alpha=opacity, color=colors[1], yerr=win_errors, label=r'$p(\text{chosen}|\text{rlhf}) - p(\text{rejected}|\text{rlhf}) < $' '\n' r'$p(\text{chosen}|\text{rlhf\_reversed}) - p(\text{rejected}|\text{rlhf\_reversed})$')

    # Labels, Title, and Custom x-axis
    ax.set_xlabel('Models')
    ax.set_ylabel('Win Rates')
    ax.set_title('RLHF Chosen vs Rejected Win Rates')
    ax.set_xticks(bar_pos)
    ax.set_xticklabels(predictions)
    ax.legend()

    plt.tight_layout()

    ax.set_ylim(-0.05, 1.05)
    plt.savefig('predictions_prompt_2.pdf')
    plt.savefig('predictions_prompt_2.png')