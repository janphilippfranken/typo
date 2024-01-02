from datasets import load_from_disk

def load_data(file_path):
    try:
        return load_from_disk(file_path)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return []

# File paths
file_paths = [
    "/scr/jphilipp/scai/experiments/hh_rlhf/constitutions/rlhf_gen_mistral_7b_instruct_eval_mistral_7b_base_run_1",
    "/scr/jphilipp/scai/experiments/hh_rlhf/constitutions/rlhf_reversed_gen_mistral_7b_instruct_eval_mistral_7b_base_run_1",
]

# Datasets 
datasets = {
    'rlhf_1': load_data(file_paths[0]),
    'rlhf_reversed_1': load_data(file_paths[1]),
}


final_constitutions = {
    'rlhf_1': [
        batch['constitutions'][-1] for batch in datasets['rlhf_1']
    ],
    'rlhf_reversed_1': [
        batch['constitutions'][-1] for batch in datasets['rlhf_reversed_1']
    ],
}

import pandas as pd

df = pd.DataFrame(final_constitutions)
df.to_csv("constitutions_1.csv")

