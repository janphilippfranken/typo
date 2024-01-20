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
    
    
    # labels form syntehtic datasets
    all_labels = {}
    for i in range(args.data.n_synthetic_labels):
        with open(f"{args.data.synthetic_data_path}/constitution_{i}_model_mixtral_7b_base.json", "r") as file:
            all_labels[i] = json.load(file)
    
    # Load and prepare dataset
    data = load_dataset(**args.data.dataset)
    dataset = data[args.data.split]

    # Load constitutions
    constitution_batch = []
    constitution_idx = 0
    for constitution_file in os.listdir(args.data.constitution_path):
        if 'synthetic' in constitution_file:
        # if 'shuffled' not in constitution_file and 'synthetic' not in constitution_file and 'reversed' not in constitution_file:
        # if 'shuffled' not in constitution_file and 'synthetic' not in constitution_file:
        # if 'shuffled' not in constitution_file:
            constitution_idx += 1
            constitution_data = load_from_disk(os.path.join(args.data.constitution_path, constitution_file))
            train_examples = constitution_data['train_examples'][0]
            labels = all_labels[int(constitution_file.split('_')[1])]['train_labels']
            example_labels = [labels[example_idx] for example_idx in train_examples]

            constitution_batch.append({
                'data': constitution_data,
                'file': constitution_file,
                'label': example_labels 
            })
    # breakpoint()
    # Process constitutions and create examples
    examples_batch = []
    n_constitutions = 0
    for constitution in tqdm(constitution_batch):
        if int(constitution['file'].split('_')[-1]) <= args.data.n_constitutions: # only take n constitutions and leave rest as test constitutions for evaluation
            n_constitutions += 1
            constitution_str = constitution['data']['constitutions'][0][-1]
            train_examples = constitution['data']['train_examples'][0]
            
            for example_idx, train_example in enumerate(train_examples):
                data_set_item_chosen = dataset[train_example]['chosen' if constitution['label'][example_idx] == 0 else 'rejected']
                data_set_item_rejected = dataset[train_example]['rejected' if constitution['label'][example_idx] == 0 else 'chosen']
                conversation, final_response_chosen = remove_final_answer(data_set_item_chosen)
                conversation, final_response_rejected = remove_final_answer(data_set_item_rejected)
                
                if 'Assistant: Human:' not in conversation:
                    examples_batch.append({
                        'prompt': f"""Write a response for the assistant that follows the principles in the constitution.\n\nAI Assistant Constitution:\n{constitution_str.strip()}\n\n{conversation.strip()}\n\nAssistant:""",
                        'chosen': final_response_chosen,
                        'rejected': final_response_rejected,
                        'label': 'chosen' if constitution['label'][example_idx] == 0 else 'rejected',
                        'constitution_file': constitution['file'],
                        'index': train_example,
                    })
                breakpoint()
    # Save formatted data
    with open(args.data.cai_data, "w") as f:
        json.dump(examples_batch, f)
        
    
    
if __name__ == "__main__":
    fire.Fire(main())