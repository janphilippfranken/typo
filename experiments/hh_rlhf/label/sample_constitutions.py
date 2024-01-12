from typing import List, Dict

import json

import fire
import numpy as np


N_CONSTITUTIONS = 10
BATCH_SIZE = 4
FILE_PATH = "principles.json"
OUTPUT_DIR = "constitutions"


def load_principles(file_path: str) -> List[Dict[str, str]]:
    with open(file_path, 'r') as file:
        return json.load(file)


def save_constitution(constitution: List, file_name: str):
    with open(file_name, 'w') as file: 
        for item in constitution[:-1]:
            file.write(item['principle'] + "\n")
        file.write(constitution[-1]['principle'])
        

def sample_principles(principles_list: List[Dict[str, str]], batch_size: int, n_constitutions: int, output_dir: str):
    for i in range(n_constitutions):
        sample = np.random.choice(principles_list, batch_size, replace=False)
        constitution = [{'principle': pair['principle']} for pair in sample]
        antithesis_constitution = [{'principle': pair['antithesis']} for pair in sample]
        constitution_combined = constitution[:batch_size//2] + antithesis_constitution[batch_size//2:]
        constitution_antithesis_combined = constitution[batch_size//2:] + antithesis_constitution[:batch_size//2]
        save_constitution(constitution_combined, f"{output_dir}/constitution_{i}.txt")
        save_constitution(constitution_antithesis_combined, f"{output_dir}/constitution_antithesis_{i}.txt")

def main():
    """Sample principles and save constitutions."""
    principles = load_principles(FILE_PATH)
    sample_principles(
        principles_list=principles, 
        batch_size=BATCH_SIZE, 
        n_constitutions=N_CONSTITUTIONS,
        output_dir=OUTPUT_DIR,
    )


if __name__ == "__main__":
    fire.Fire(main())