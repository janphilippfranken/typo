import json
import seaborn as sns
import os
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

FILE_NAME = "constitutions-35-40-examples-700-800.json"  # File with the examples data

def load_examples(filename):
    with open(filename, 'r') as file:
        data = json.load(file)
    return data

def plot_labels(examples):
    constitution_ids = [e['constitution_id'] for e in examples]
    labels = [1 if e['label'] == 'rejected' else 0 for e in examples]

    # Calculate average label for each constitution
    label_sums = defaultdict(int)
    counts = defaultdict(int)

    for cid, label in zip(constitution_ids, labels):
        label_sums[cid] += label
        counts[cid] += 1

    averages = {cid: label_sums[cid] / counts[cid] for cid in label_sums}

    # Sorting by constitution ID for consistent plotting
    sorted_ids = sorted(averages.keys())
    sorted_averages = [averages[cid] for cid in sorted_ids]

    # Plotting the bar plot
    plt.figure(figsize=(10, 5))
    sns.set_theme(style="darkgrid")
    plt.bar(sorted_ids, sorted_averages)
    plt.ylim(-0.05, 1.05)
    plt.xlabel("Constitution ID")
    plt.yticks([0, .2, .4, .6, .8, 1.0])
    plt.ylabel("Percentage label = 1 (Rejected)")
    plt.savefig(f"{FILE_NAME.split('.json')[0]}_bar.png")

    # Create and plot the heatmap
    unique_ids = list(set(constitution_ids))
    unique_ids.extend(['hh', 'hh_flipped'])  # Add special constitutions
    agreement_matrix = np.zeros((len(unique_ids), len(unique_ids)))

    for i, id1 in enumerate(unique_ids):
        for j, id2 in enumerate(unique_ids):
            if id1 in ['hh', 'hh_flipped'] and id2 in ['hh', 'hh_flipped']:
                const_labels_1 = [0 if id1 == 'hh'else 1]
                const_labels_2 = [0 if id2 == 'hh'else 1]
                agreement = np.mean([l1 == l2 for l1, l2 in zip(const_labels_1, const_labels_2)])
            elif id1 in ['hh', 'hh_flipped'] or id2 in ['hh', 'hh_flipped']:
                const_labels = [0 if id1 == 'hh' or id2 == 'hh' else 1]
                other_labels = [label for cid, label in zip(constitution_ids, labels) if cid == id1 or cid == id2]
                agreement = np.mean([l1 == l2 for l1, l2 in zip(const_labels * len(other_labels), other_labels)])
            else:
                labels1 = [label for cid, label in zip(constitution_ids, labels) if cid == id1]
                labels2 = [label for cid, label in zip(constitution_ids, labels) if cid == id2]
                agreement = np.mean([l1 == l2 for l1, l2 in zip(labels1, labels2)])
            agreement_matrix[i, j] = agreement

    plt.figure(figsize=(12, 12))
    sns.heatmap(agreement_matrix, annot=True, xticklabels=unique_ids, yticklabels=unique_ids, cmap='viridis')
    plt.title("Agreement between Constitutions")
    plt.savefig(f"{FILE_NAME.split('.json')[0]}_heatmap.png")

def main():
    examples = load_examples(FILE_NAME)
    plot_labels(examples)

if __name__ == "__main__":
    main()