import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

BATCH_SIZE = 10

def load_data(file_path):
    try:
        with open(file_path, "r") as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return []

# File paths
file_paths = [
    "predictions/rlhf_gen_mistral_7b_instruct_eval_mistral_7b_base_run_1_model_mixtral_7b_instruct.json",
    "predictions/rlhf_reversed_gen_mistral_7b_instruct_eval_mistral_7b_base_run_1_model_mixtral_7b_instruct.json",
    "predictions/rlhf_gen_mistral_7b_instruct_eval_mistral_7b_base_run_2_model_mixtral_7b_instruct.json",
    "predictions/rlhf_reversed_gen_mistral_7b_instruct_eval_mistral_7b_base_run_2_model_mixtral_7b_instruct.json",
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
    mean = np.mean(values)
    error = 1.95 * (np.std(values) / np.sqrt(len(values)))
    return mean, error

chosen_means, chosen_errors = zip(*(calculate_stats([data[i]["chosen"] for i in data]) for data in datasets))
rejected_means, rejected_errors = zip(*(calculate_stats([data[i]["rejected"] for i in data]) for data in datasets))
n_evaluated_means = [np.mean([data[i]["n_evaluated"] for i in data])/BATCH_SIZE for data in datasets]

# Dataset names
predictions = [
    f"rlhf-run-1 (N {len(datasets[0])})", 
    f"rlhf-reversed-run-1 (N {len(datasets[1])})",
    f"rlhf-run-2 (N {len(datasets[2])})", 
    f"rlhf-reversed-run-2 (N {len(datasets[3])})",
]

# Plotting
fig, ax = plt.subplots(figsize=(10, 5))
bar_width = 0.2
opacity = 0.8

# Bar positions
bar_pos_chosen = np.arange(len(predictions))
bar_pos_rejected = [x + bar_width for x in bar_pos_chosen]
bar_pos_eval = [x + bar_width for x in bar_pos_rejected]
 
# Bars for Chosen
ax.bar(bar_pos_chosen, chosen_means, bar_width, alpha=opacity, color=colors[0], yerr=chosen_errors, label='% Chosen')

# Bars for Rejected
ax.bar(bar_pos_rejected, rejected_means, bar_width, alpha=opacity, color=colors[1], yerr=rejected_errors, label='% Rejected')

# Bars for N Evaluated
ax.bar(bar_pos_eval, n_evaluated_means, bar_width, alpha=opacity, color=colors[2], label='% Evaluated')

# Labels, Title and Custom x-axis
ax.set_xlabel('Dataset')
ax.set_ylabel('Percentage')
ax.set_title('Final Assistant Response on HH-RLHF (test split)')
ax.set_xticks([r + bar_width for r in range(len(predictions))])
ax.set_xticklabels(predictions)
ax.legend()

ax.set_ylim(-0.05, 1.05)

# Show plot
plt.tight_layout()
plt.savefig('predictions.pdf')
plt.savefig('predictions.png')
