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

    
    
def calculate_win_rate_and_error(chosen_1, chosen_2, rejected_1, rejected_2):
    
    B, N = chosen_1.shape
    win_rates = np.zeros((B, N))
    
    for batch_idx in range(B):
        diff_1 = chosen_1[batch_idx] - chosen_2[batch_idx]
        diff_2 = rejected_1[batch_idx] - rejected_2[batch_idx]
        win_rates[batch_idx] = (diff_1 > diff_2).astype(int)
    
    win_rate_means = np.mean(win_rates, axis=-1)
    win_rate_errors = 1.96 * np.sqrt(np.mean(win_rates, axis=-1) * (1 - np.mean(win_rates, axis=-1)) / N) 

    return win_rate_means, win_rate_errors

def main(args):
    for run in range(args.start, args.n_runs):
        file_paths = [
            f"predictions/rlhf_gen_mixtral_7b_base_eval_mixtral_7b_base_gen_prompt_generation_prompt_base_2_run_{run}_model_mixtral_7b_base_train_n_final_{args.n_final}.json",
            f"predictions/rlhf_reversed_gen_mixtral_7b_base_eval_mixtral_7b_base_gen_prompt_generation_prompt_base_2_run_{run}_model_mixtral_7b_base_train_n_final_{args.n_final}.json",
            f"predictions/rlhf_gen_mixtral_7b_base_eval_mixtral_7b_base_gen_prompt_generation_prompt_base_2_run_{run}_model_mixtral_7b_base_test_n_final_{args.n_final}.json",
            f"predictions/rlhf_reversed_gen_mixtral_7b_base_eval_mixtral_7b_base_gen_prompt_generation_prompt_base_2_run_{run}_model_mixtral_7b_base_test_n_final_{args.n_final}.json",
        ]

        # Loading data
        datasets = [load_data(path) for path in file_paths]


        # Setting the theme and font
        sns.set_theme(style="darkgrid")
        plt.rcParams['font.family'] = 'DejaVu Sans'
        plt.figure(figsize=(10, 5))
        colors = sns.palettes.color_palette("colorblind", 10)


        def get_probs(dataset):
            probs_chosen = np.zeros((len(dataset), len(dataset['0'])))
            probs_rejected = np.zeros((len(dataset), len(dataset['0'])))
            for batch_idx, examples in enumerate(dataset.values()):
                for example_idx, example_key in enumerate(examples):
                    probs_chosen[batch_idx, example_idx] = examples[example_key]["chosen"]
                    probs_rejected[batch_idx, example_idx] = examples[example_key]["rejected"]
            return probs_chosen, probs_rejected


        probs = [get_probs(dataset) for dataset in datasets]
        
        
        win_rates_train, errors_train = calculate_win_rate_and_error(
            probs[0][0], probs[1][0], probs[0][1], probs[1][1]
        )

        win_rates_test, errors_test = calculate_win_rate_and_error(
            probs[2][0], probs[3][0], probs[2][1], probs[3][1]
        )

        # Combine win rates and errors
        win_rates_chosen = [np.mean(win_rates_train), np.mean(win_rates_test)]
        win_rates_rejected = [1 - np.mean(win_rates_train), 1 - np.mean(win_rates_test)]
        win_errors = [errors_train, errors_test]
        
        print(win_rates_chosen, win_rates_rejected)

        win_errors_batch = [
            1.96 * np.sqrt(np.mean(win_rates_train) * (1 - np.mean(win_rates_train)) / win_rates_train.shape[0]),
            1.96 * np.sqrt(np.mean(win_rates_test) * (1 - np.mean(win_rates_test)) / win_rates_test.shape[0]),
        ]

        # Names for x-axis labels
        predictions = [
            f"train split (N {probs[0][0].shape[1]})", 
            f"test split (N {probs[2][0].shape[1]})", 
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
        label_1 = 'hh_rlhf'
        label_2 = 'hh_rlhf_flipped'
        
        if label_2 != 'hh_rlhf_shuffled':

            ax.bar(bar_pos_chosen, win_rates_chosen, bar_width, alpha=opacity, color=colors[0], yerr=win_errors,
                label=f'$p(\\text{{chosen}}|\\text{{{label_1}}}) - p(\\text{{chosen}}|\\text{{{label_2}}}) > $'
                        f'\n$p(\\text{{rejected}}|\\text{{{label_1}}}) - p(\\text{{rejected}}|\\text{{rlhf_flipped}})$')

            ax.bar(bar_pos_rejected, win_rates_rejected, bar_width, alpha=opacity, color=colors[1], yerr=win_errors,
                label=f'$p(\\text{{chosen}}|\\text{{{label_1}}}) - p(\\text{{chosen}}|\\text{{{label_2}}}) < $'
                        f'\n$p(\\text{{rejected}}|\\text{{{label_1}}}) - p(\\text{{rejected}}|\\text{{rlhf_flipped}})$')
            # Labels, Title, and Custom x-axis
            
            ax.set_xticks(bar_pos)
            ax.set_xticklabels(predictions)


        else:
            win_rates_chosen[1] = 0
            win_errors[1] = 0
            win_rates_rejected[1] = 0
            win_errors[1] = 0
            
            ax.bar(bar_pos_chosen, win_rates_chosen, bar_width, alpha=opacity, color=colors[0], yerr=win_errors,
                label=f'$p(\\text{{chosen}}|\\text{{{label_1}}}) - p(\\text{{chosen}}|\\text{{{label_2}}}) > $'
                        f'\n$p(\\text{{rejected}}|\\text{{{label_1}}}) - p(\\text{{rejected}}|\\text{{rlhf_flipped}})$')

            ax.bar(bar_pos_rejected, win_rates_rejected, bar_width, alpha=opacity, color=colors[1], yerr=win_errors,
                label=f'$p(\\text{{chosen}}|\\text{{{label_1}}}) - p(\\text{{chosen}}|\\text{{{label_2}}}) < $'
                        f'\n$p(\\text{{rejected}}|\\text{{{label_1}}}) - p(\\text{{rejected}}|\\text{{rlhf_flipped}})$')
            # Labels, Title, and Custom x-axis
            
            ax.set_xticks(bar_pos)
            ax.set_xticklabels(predictions)
            

            # Labels, Title, and Custom x-axis
        # ax.set_xlabel('Models')
        ax.set_ylabel('Win Rates')
        ax.set_title('RLHF Chosen vs Rejected Win Rates')
        ax.legend()

        plt.tight_layout()

        ax.set_ylim(-0.05, 1.05)
       
        plt.savefig(f'./plots/mixtral_run_{run}_hh_rlhf_vs_flipped.pdf')
        plt.savefig(f'./plots/mixtral_run_{run}_hh_rlhf_vs_flipped.png')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--n_runs', default=51, type=int, help='Number of runs to process')
    parser.add_argument('--start',  default=1, type=int, help='Start')
    parser.add_argument('--n_final',  default=1, type=int, help='Start')
    args = parser.parse_args()
    main(args)