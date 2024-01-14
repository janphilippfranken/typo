import json
import numpy as np
import os

N_CONSTITUTIONS = 5
BATCH_SIZE = 4
FILE_PATH = "principles.json"
OUTPUT_DIR = "constitutions"

def load_principles(file_path: str):
    with open(file_path, 'r') as file:
        return json.load(file)

def save_constitution(constitution, file_name):
    with open(file_name, 'w') as file:
        for item in constitution[:-1]:
            file.write(item['principle'] + "\n")
        file.write(constitution[-1]['principle'])

def sample_principles(principles_list, batch_size, n_constitutions, output_dir):
    all_constitutions = []

    for _ in range(n_constitutions):
        while True:
            sample = np.random.choice(principles_list, batch_size, replace=False)
            constitution = [{'principle': pair['principle']} for pair in sample]
            antithesis_constitution = [{'principle': pair['antithesis']} for pair in sample]

            # Combine constitution and antithesis principles
            constitution_combined = constitution[:batch_size//2] + antithesis_constitution[batch_size//2:]
            constitution_antithesis_combined = constitution[batch_size//2:] + antithesis_constitution[:batch_size//2]

            # Check for overlap
            if all(len(set(const['principle'] for const in constitution_combined) & set(prev_const['principle'] for prev_const in prev_constitution)) <= batch_size // 2 for prev_constitution in all_constitutions):
                all_constitutions.append(constitution_combined)
                break

        # save_constitution(constitution_combined, os.path.join(output_dir, f"constitution_{len(all_constitutions)-1}.txt"))
        # save_constitution(constitution_antithesis_combined, os.path.join(output_dir, f"constitution_antithesis_{len(all_constitutions)-1}.txt"))

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    principles = load_principles(FILE_PATH)
    sample_principles(principles, BATCH_SIZE, N_CONSTITUTIONS, OUTPUT_DIR)

if __name__ == "__main__":
    main()