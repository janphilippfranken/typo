import json
import numpy as np
import os
from tqdm import tqdm

# Settings
PRINCIPLES_DIR = "principles"
OUTPUT_DIR = "constitutions"
CONSTITUTION_DIMENSIONS = {
    "h": ["helpful"],
    "hh": ["helpful", "harmless"],
    # "hhh": ["helpful", "harmless", "honest"],
    # "hhhh": ["helpful", "harmless", "honest", "humor"],
    # "hhhhh": ["helpful", "harmless", "honest", "humor", "heedful"],
}
N_CONSTITUTIONS_PER_DIMENSION = 25
np.random.seed(1)  # For reproducibility

def load_principles(category):
    file_path = os.path.join(PRINCIPLES_DIR, f"{category}.json")
    with open(file_path, 'r') as file:
        return json.load(file)

def save_constitution(constitution, dimension, index, is_anticonstitution=False):
    suffix = "_anti" if is_anticonstitution else ""
    file_path = os.path.join(OUTPUT_DIR, f"{dimension}{suffix}_{index}.json")
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as file:
        json.dump({"principles": constitution}, file, indent=4)

def generate_constitutions(dimensions, n_constitutions):
    # Initialize a dictionary to keep track of selected principles for each index
    index_selected_principles = {index: {} for index in range(n_constitutions)}

    for dimension_key, categories in dimensions.items():
        for index in tqdm(range(n_constitutions)):
            constitution = []
            anticonstitution = []
            
            for category in categories:
                principles = load_principles(category)
                
                # Check if a principle has already been selected for this index and category
                if category in index_selected_principles[index]:
                    selected = index_selected_principles[index][category]
                else:
                    # Select a new principle and remember it
                    selected = np.random.choice(principles)
                    index_selected_principles[index][category] = selected
                
                constitution.append(selected['definition'])
                
                # For anticonstitutions, the last category's principle is replaced with its antithesis
                if category == categories[-1]:
                    anticonstitution.append(selected['antithesis'])
                else:
                    anticonstitution.append(selected['definition'])

            # Save the constitution and its anticonstitution
            save_constitution(constitution, dimension_key, index)
            save_constitution(anticonstitution, dimension_key, index, is_anticonstitution=True)

if __name__ == "__main__":
    generate_constitutions(CONSTITUTION_DIMENSIONS, N_CONSTITUTIONS_PER_DIMENSION)
