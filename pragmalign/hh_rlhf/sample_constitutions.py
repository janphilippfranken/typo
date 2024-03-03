import json
import numpy as np
import os
from tqdm import tqdm

# Settings
PRINCIPLES_DIR = "principles/test"
OUTPUT_DIR = "constitutions/test"
CONSTITUTION_DIMENSIONS = {
    "hh": ["helpful", "harmless"],
}
N_PRINCIPLES_PER_DIMENSION = 5  # Assuming 10 principles per category
np.random.seed(1)  # For reproducibility

def load_principles(category):
    file_path = os.path.join(PRINCIPLES_DIR, f"{category}.json")
    with open(file_path, 'r') as file:
        return json.load(file)

def save_constitution(constitution, dimension, constitution_type, index):
    file_path = os.path.join(OUTPUT_DIR, f"{dimension}_{constitution_type}_{index}.json")
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as file:
        json.dump({"principles": constitution}, file, indent=4)

def generate_constitutions(dimensions):
    for dimension_key, categories in dimensions.items():
        helpful_principles = load_principles("helpful")
        harmless_principles = load_principles("harmless")

        index = 0  # Initialize a counter for unique filename generation
        for helpful in [True, False]:
            for harmless in [True, False]:
                for helpful_index in tqdm(range(N_PRINCIPLES_PER_DIMENSION), desc=f"Generating {dimension_key} - {'H' if helpful else 'NH'}{'H' if harmless else 'NH'}"):
                    for harmless_index in tqdm(range(N_PRINCIPLES_PER_DIMENSION), leave=False):
                        constitution_type = f"{'h' if helpful else 'not_h'}_{'h' if harmless else 'not_h'}"
                        constitution = [
                            helpful_principles[helpful_index]['definition'] if helpful else helpful_principles[helpful_index]['antithesis'],
                            harmless_principles[harmless_index]['definition'] if harmless else harmless_principles[harmless_index]['antithesis']
                        ]
                        save_constitution(constitution, dimension_key, constitution_type, index)
                        index += 1  # Increment the counter for each saved constitution

if __name__ == "__main__":
    generate_constitutions(CONSTITUTION_DIMENSIONS)