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


def main():
    
    file_paths = [
        f"labels/constitution_0_model_mixtral_7b_base.json",
        f"labels/constitution_1_model_mixtral_7b_base.json",
    ]

    # Loading data
    datasets = [load_data(path) for path in file_paths]


    # Setting the theme and font
    sns.set_theme(style="darkgrid")
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.figure(figsize=(10, 5))
    colors = sns.palettes.color_palette("colorblind", 10)

    
    differences_train = (np.array(datasets[0]['train']) != np.array(datasets[1]['train'])).astype(int)
    differences_train_error = 1.96 * np.sqrt(np.mean(differences_train, axis=-1) * (1 - np.mean(differences_train, axis=-1)) / len(differences_train)) 
    differences_test = (np.array(datasets[0]['test']) != np.array(datasets[1]['test'])).astype(int)
    differences_test_error = 1.96 * np.sqrt(np.mean(differences_test, axis=-1) * (1 - np.mean(differences_test, axis=-1)) / len(differences_test)) 

    # Combine win rates and errors
    agreement =  [1- np.mean(differences_train), 1- np.mean(differences_test)]
    disagreement = [np.mean(differences_train), np.mean(differences_test)]
    win_errors = [differences_train_error, differences_test_error]

    # Names for x-axis labels
    predictions = [
        f"train split", 
        f"test split", 
    ]

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
    label_1 = 'one'
    label_2 = 'two'
    

    ax.bar(bar_pos_chosen, agreement, bar_width, alpha=opacity, color=colors[0], yerr=win_errors,
        label=f'$agree$')

    ax.bar(bar_pos_rejected, disagreement, bar_width, alpha=opacity, color=colors[1], yerr=win_errors,
        label=f'disagree')
  
    ax.set_xticks(bar_pos)
    ax.set_xticklabels(predictions)


    # ax.set_xlabel('Models')
    ax.set_ylabel('Win Rates')
    ax.legend()

    plt.tight_layout()

    ax.set_ylim(-0.05, 1.05)
    plt.show()
    plt.savefig(f'test.pdf')
    plt.savefig(f'test.png')

if __name__ == "__main__":
   
    main()