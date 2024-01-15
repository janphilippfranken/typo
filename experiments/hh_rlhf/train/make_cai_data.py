import logging
import os 
import fire
import json
import hydra
from tqdm import tqdm
from omegaconf import DictConfig

from datasets import load_dataset, load_from_disk

from helpers import remove_final_answer


@hydra.main(version_base=None, config_path="conf", config_name="make_cai_data")
def main(args: DictConfig) -> None:
    
    
    # Load and prepare dataset
    data = load_dataset(**args.data.dataset)
    dataset = data[args.data.split]

    # Load constitutions
    constitution_batch = []
    constitution_idx = 0
    for constitution_file in os.listdir(args.data.constitution_path):
        if 'shuffled' not in constitution_file:
            constitution_idx += 1
            constitution_data = load_from_disk(os.path.join(args.data.constitution_path, constitution_file))
            label = [1] * args.data.n_examples if 'reversed' in constitution_file else [0] * args.data.n_examples # TODO: include option to have synthetic labels
            constitution_batch.append({
                'data': constitution_data,
                'file': constitution_file,
                'label': label
            })
    print(len(constitution_batch))
    # Process constitutions and create examples
    examples_batch = []
    for constitution in tqdm(constitution_batch):
        constitution_str = constitution['data']['constitutions'][0][-1]
        train_examples = constitution['data']['train_examples'][0]
        
        for example_idx, train_example in enumerate(train_examples):
            data_set_item = dataset[train_example]['chosen' if constitution['label'][example_idx] == 0 else 'rejected']
            conversation, final_response = remove_final_answer(data_set_item)
            if 'Assistant: Human:' not in conversation:
                examples_batch.append({
                    'constitution': constitution_str,
                    'conversation': conversation,
                    'final_response': final_response,
                    'label': 'chosen' if constitution['label'][example_idx] == 0 else 'rejected',
                    'constitution_file': constitution['file'],
                    'index': train_example,
                })
    print(len(examples_batch))
    # Save formatted data
    with open(args.data.cai_data, "w") as f:
        json.dump(examples_batch, f)
        
    
    
if __name__ == "__main__":
    fire.Fire(main())