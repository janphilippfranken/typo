import json
import logging


import fire
import hydra
import numpy as np
from tqdm import tqdm
from omegaconf import DictConfig
from datasets import load_dataset


from scaituning.models.vllm_models.inference_model import HFInferenceModel

from helpers import *

logging.basicConfig(level=logging.INFO)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(args: DictConfig) -> None:

    # GET MODEL(S)    
    model = VLLMInferenceModel(**args.model.model_config)
   

    # GET PRINCIPLES
    dataset = load_dataset(**args.data.dataset)
    dataset_train = dataset['train']
    dataset_test = dataset['test']
    
    
    # RESULTS DICT (IE WHICH LABELS ARE USED)
    results = {
        k: {
            'train_labels': [],
            'train_logprobs': [],
            'train_probs': [],
            
            'test_labels': [],
            'test_logprobs': [],
            'test_probs': [],
        } 
        for k in range(args.label.n_constitutions)
    }
    
    constitution_strings = []
    
    # MAIN LOOP
    for constitution_idx in tqdm(range(args.label.n_constitutions)):

        # LOAD CONSTITUTION 
        with open(f"{args.label.constitutions}/constitution_{constitution_idx}.txt", 'r') as file: 
            constitution = file.readlines()

            
        constitution_str = "" 
        for i, principle in enumerate(constitution): 
            constitution_str += f"{i + 1}. {principle}"
            
        constitution_strings.append(constitution_str)
            
    
    # EXAMPLES FOR LABELLING
    examples_train = range(args.label.n_examples_train)
    examples_test = range(args.label.n_examples_test)


    # TRAIN LABELS
    for example_idx in tqdm(examples_train): 

        log_prob_chosen_constitution, log_prob_rejected_constitution = run_eval_mcq(
            dataset=dataset_train,
            constitutions=constitution_strings,
            model=model,
            eval_prompt=args.label.evaluation_prompt,
            example_idx=example_idx,
        )
        
        for constitution_idx in range(args.label.n_constitutions):
                
            probs = np.array(
                [
                    float(log_prob_chosen_constitution[constitution_idx]), 
                    float(log_prob_rejected_constitution[constitution_idx]),
                ]
            )
            probs = np.exp(probs) / np.sum(np.exp(probs))
            
            label = np.argmax(probs)
            
            results[constitution_idx]['train_labels'].append(int(label))
            results[constitution_idx]['train_logprobs'].append(
                [
                    float(log_prob_chosen_constitution[constitution_idx]), 
                    float(log_prob_rejected_constitution[constitution_idx]),
                ]
            )
            results[constitution_idx]['train_probs'].append(probs[0])
        
            # Save progress after each example
            with open(f"{args.label.storage_path}/constitution_{constitution_idx}_model_{args.model.name}.json", "w") as f:
                json.dump(results[constitution_idx], f)
            
    
    # TEST LABELS
    for example_idx in tqdm(examples_test): 
        
        log_prob_chosen_constitution, log_prob_rejected_constitution = run_eval_mcq(
            dataset=dataset_test,
            constitutions=constitution_strings,
            model=model,
            eval_prompt=args.label.evaluation_prompt,
            example_idx=example_idx,
        )
        
        for constitution_idx in range(args.label.n_constitutions):
                
            probs = np.array(
                [
                    float(log_prob_chosen_constitution[constitution_idx]), 
                    float(log_prob_rejected_constitution[constitution_idx]),
                ]
            )
            probs = np.exp(probs) / np.sum(np.exp(probs))
            
            label = np.argmax(probs)
            
            results[constitution_idx]['test_labels'].append(int(label))
            results[constitution_idx]['test_logprobs'].append(
                [
                    float(log_prob_chosen_constitution[constitution_idx]), 
                    float(log_prob_rejected_constitution[constitution_idx]),
                ]
            )
            results[constitution_idx]['test_probs'].append(probs[0])
        
            # Save progress after each example
            with open(f"{args.label.storage_path}/constitution_{constitution_idx}_model_{args.model.name}.json", "w") as f:
                json.dump(results[constitution_idx], f)
        
        
if __name__ == '__main__':
    fire.Fire(main())