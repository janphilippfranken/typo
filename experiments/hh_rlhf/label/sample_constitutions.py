import json
import numpy as np
import os

N_CONSTITUTIONS = 2000
BATCH_SIZE = 8
FILE_PATH = "synthetic_principles.json"
OUTPUT_DIR = "synthetic-constitutions"

np.random.seed(1)

def load_principles(file_path: str):
    with open(file_path, 'r') as file:
        return json.load(file)

def save_constitution(constitution, file_name, constitution_id):
    constitution_with_id = {
        "constitution_id": str(constitution_id),
        "constitution": "\n".join(f"{i + 1}. {item['principle']}" for i, item in enumerate(constitution))
    }
    with open(file_name, 'w') as file:
        json.dump(constitution_with_id, file, indent=4)

def get_unique_constitution_key(constitution):
    return tuple(sorted(item['principle'] for item in constitution))

def sample_principles(principles_list, n_constitutions, output_dir):
    unique_constitutions = set()
    for i in range(n_constitutions):
        while True:
            principles_sample = np.random.choice(principles_list, BATCH_SIZE // 2, replace=False)
            principles = [{'principle': pair['principle']} for pair in principles_sample]

            available_antitheses = [p for p in principles_list if p not in principles_sample]
            antitheses_sample = np.random.choice(available_antitheses, BATCH_SIZE // 2, replace=False)
            antitheses = [{'principle': pair['antithesis']} for pair in antitheses_sample]

            constitution = principles + antitheses
            np.random.shuffle(constitution)

            constitution_key = get_unique_constitution_key(constitution)
            
            if constitution_key not in unique_constitutions:
                unique_constitutions.add(constitution_key)
                break

        save_constitution(constitution, os.path.join(output_dir, f"constitution_{i}.json"), i)

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    principles = load_principles(FILE_PATH)
    sample_principles(principles, N_CONSTITUTIONS, OUTPUT_DIR)

if __name__ == "__main__":
    main()