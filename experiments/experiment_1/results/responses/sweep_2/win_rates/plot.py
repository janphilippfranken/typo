import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set the aesthetic style of the plots
sns.set_style("whitegrid")

# Get list of JSON files in the current directory
json_files = [f for f in os.listdir('.') if f.endswith('.json')]

epochs = [0.08, 0.17, 0.25, 0.34, 0.42, 0.5, 0.59, 0.67, 0.76, 0.84, 0.92, 1.0]
iteration = "0vsbase"

# Initialize lists to store average rates for each epoch
helpful_rates = []
harmless_rates = []
epoch_labels = []

# Iterate over epochs and read the corresponding JSON files for helpful and harmless rates
for epoch in epochs:
    print(epoch)
    helpful_file = f'typo-iteration-{iteration}-lr-1e-7-beta-0.0-epoch-{epoch:.2f}-temperature-0.0-helpful.json'
    harmless_file = f'typo-iteration-{iteration}-lr-1e-7-beta-0.0-epoch-{epoch:.2f}-temperature-0.0-harmless.json'

    if helpful_file in json_files:
        with open(helpful_file, 'r') as f:
            data = json.load(f)
            helpful_rates.append(np.mean(data))
            epoch_labels.append(epoch)
    else:
        helpful_rates.append(None)

    if harmless_file in json_files:
        with open(harmless_file, 'r') as f:
            data = json.load(f)
            harmless_rates.append(np.mean(data))
    else:
        harmless_rates.append(None)
    print(helpful_rates)

# Prepare the data for plotting
data = {
    'Epoch': epoch_labels,
    'Helpful Win Rate': [rate for rate in helpful_rates if rate is not None],
    'Harmless Win Rate': [rate for rate in harmless_rates if rate is not None]
}

# Convert the data to a pandas DataFrame
import pandas as pd
df = pd.DataFrame(data)

# Plot
fig, axs = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

sns.barplot(x='Epoch', y='Helpful Win Rate', data=df, ax=axs[0], palette='Blues_d')
axs[0].set_title('Helpful Win Rates')
axs[0].set_ylabel('Win Rate')

sns.barplot(x='Epoch', y='Harmless Win Rate', data=df, ax=axs[1], palette='Greens_d')
axs[1].set_title('Harmless Win Rates')
axs[1].set_ylabel('')

plt.tight_layout()
plt.savefig('win_rates_2vs1.png')
plt.show()
