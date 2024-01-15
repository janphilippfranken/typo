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

def compute_agreement(dataset1, dataset2):
    differences_train = (np.array(dataset1['train_labels']) != np.array(dataset2['train_labels'])).astype(int)
    differences_test = (np.array(dataset1['test_labels']) != np.array(dataset2['test_labels'])).astype(int)
    agreement_train = 1 - np.mean(differences_train)
    agreement_test = 1 - np.mean(differences_test)
    return agreement_train, agreement_test

def main(n):
    # Generate file paths
    file_paths = [f"labels/constitution_{i}_model_mixtral_7b_base.json" for i in range(n)]

    # Loading data
    datasets = [load_data(path) for path in file_paths]
    breakpoint()
    datasets = [
        {
            'train_labels': np.repeat(0, len(datasets[0]['train_labels'])),
            'test_labels': np.repeat(0, len(datasets[0]['test_labels'])),
        }
    ] + [
        {
            'train_labels': np.repeat(1, len(datasets[0]['train_labels'])),
            'test_labels': np.repeat(1, len(datasets[0]['test_labels'])),
        }
    ] +  datasets
    breakpoint()
    print(len(datasets))
    # Compute agreement matrix
    agreement_matrix = np.zeros((2, n + 2, n + 2))
    for i in range(n + 2):
        for j in range(n + 2):
            if i != j:
                agreement_matrix[0][i][j] = compute_agreement(datasets[i], datasets[j])[0]
                agreement_matrix[1][i][j] = compute_agreement(datasets[i], datasets[j])[1]
            else:
                agreement_matrix[0][i][j] = compute_agreement(datasets[i], datasets[j])[0]
                agreement_matrix[1][i][j] = compute_agreement(datasets[i], datasets[j])[1]

    # Plotting
    sns.set_theme(style="darkgrid")
    plt.figure(figsize=(10, 10))
    ax = sns.heatmap(agreement_matrix[0], annot=True, cmap="viridis", square=True)
    ax.set_xlabel('Constitution Index')
    ax.set_ylabel('Constitution Index')
    plt.title('Average Agreement Rates')
    plt.tight_layout()
    plt.savefig("train.png")
    plt.show()

    plt.figure(figsize=(10, 10))
    ax = sns.heatmap(agreement_matrix[1], annot=True, cmap="viridis", square=True)
    ax.set_xlabel('Constitution Index')
    ax.set_ylabel('Constitution Index')
    plt.title('Average Agreement Rates')
    plt.tight_layout()
    plt.savefig("test.png")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process number of files.')
    parser.add_argument('--n', type=int, default=7, help='Number of files to process')
    args = parser.parse_args()
    main(args.n)
