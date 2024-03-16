import os
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Directory containing the JSON files (assuming the current directory for demonstration)
directory = "win_rates"

# Check if the directory exists to avoid FileNotFoundError
if not os.path.exists(directory):
    os.makedirs(directory)

# Prepare data structures to hold the win rates
win_rates = {'helpful': [], 'harmless': []}
epoch_values = np.linspace(0.1, 1.0, 10)  # Epochs from 0.1 to 1.0 in 10 steps

# Loop over all json files and calculate the mean win rates
for filename in sorted(os.listdir(directory)):
    if filename.endswith(".json"):
        # Extract the epoch and type (helpful or harmless) from the filename
        parts = filename.split('-')
        epoch_str = parts[2]
        file_type = parts[-1].split('.')[0]  # 'helpful' or 'harmless'
        # breakpoint()

        # Parse the epoch number as float to match against epoch_values
        epoch_num = float(epoch_str.split('-')[-1])

        # Construct the full file path
        file_path = os.path.join(directory, filename)
        with open(file_path, 'r') as file:
            # Load the data from the json file and calculate the mean win rate
            data = json.load(file)
            win_rate = np.mean(data)

            # Append the win rate to the corresponding list based on type
            if epoch_num in epoch_values:
                win_rates[file_type].append((epoch_num, win_rate))

# Convert to DataFrame for plotting
df_list = []
for file_type, values in win_rates.items():
    for epoch, rate in values:
        df_list.append({'Epoch': epoch, 'Win Rate': rate, 'Type': file_type.capitalize()})
df = pd.DataFrame(df_list)

# Plotting with seaborn
sns.set(style="whitegrid")
plt.figure(figsize=(14, 7))
barplot = sns.barplot(x="Epoch", y="Win Rate", hue="Type", data=df)
barplot.set_title('Win Rates by Epoch')
plt.legend(title='Type')
plt.tight_layout()

# Save the figure
plot_path = "win_rates/win_rates_barplot.png"
plt.savefig(plot_path)