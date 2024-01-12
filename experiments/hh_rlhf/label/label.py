import json
import logging


import fire
import hydra
import numpy as np
from tqdm import tqdm
from omegaconf import DictConfig
from datasets import load_dataset


from scaituning.models.vllm_models.inference_model import VLLMInferenceModel

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
            'train': [],
            'test': []} 
        for k in range(args.label.n_constitutions)
    }
    
    
    # MAIN LOOP
    for constitution_idx in tqdm(range(args.label.n_constitutions)):

        # LOAD CONSTITUTION 
        with open(f"{args.label.constitutions}/constitution_{constitution_idx}.txt", 'r') as file: 
            constitution = file.readlines()
            
        with open(f"{args.label.constitutions}/constitution_antithesis_{constitution_idx}.txt", 'r') as file: 
            constitution_antithesis = file.readlines()
        
        constitution_str = "" 
        for principle in constitution: 
            constitution_str += principle
            
        constitution_antithesis_str = "" 
        for principle in constitution_antithesis: 
            constitution_antithesis_str += principle
     
        # EXAMPLES FOR LABELLING
        examples_train = range(args.label.n_examples_train)
        examples_test = range(args.label.n_examples_test)


        for example_idx in tqdm(examples_train): 
            
            log_prob_chosen_constitution, log_probs_rejected_constitution = run_eval_log_probs(
                dataset=dataset_train,
                constitution=constitution_str,
                model=model,
                eval_prompt=args.label.evaluation_prompt,
                example_idx=example_idx,
            )
            
            log_prob_chosen_antithesis, log_probs_rejected_antithesis = run_eval_log_probs(
                dataset=dataset_train,
                constitution=constitution_antithesis_str,
                model=model,
                eval_prompt=args.label.evaluation_prompt,
                example_idx=example_idx,
            )
            
            # COMPUTE PROBS
            label = log_prob_chosen_constitution - log_prob_chosen_antithesis > log_probs_rejected_constitution - log_probs_rejected_antithesis
            results[constitution_idx]['train'].append(int(label))
            
            
        for example_idx in tqdm(examples_test): 
            
            log_prob_chosen_constitution, log_probs_rejected_constitution = run_eval_log_probs(
                dataset=dataset_test,
                constitution=constitution_str,
                model=model,
                eval_prompt=args.label.evaluation_prompt,
                example_idx=example_idx,
            )
            
            log_prob_chosen_antithesis, log_probs_rejected_antithesis = run_eval_log_probs(
                dataset=dataset_test,
                constitution=constitution_antithesis_str,
                model=model,
                eval_prompt=args.label.evaluation_prompt,
                example_idx=example_idx,
            )
            
            # COMPUTE PROBS
            label = log_prob_chosen_constitution - log_prob_chosen_antithesis > log_probs_rejected_constitution - log_probs_rejected_antithesis
            results[constitution_idx]['test'].append(int(label))
            
       
        # WRITE TO JSON
        with open(f"{args.label.storage_path}/constitution_{constitution_idx}_model_{args.model.name}.json", "w") as f:
            json.dump(results[constitution_idx], f)

        
if __name__ == '__main__':
    fire.Fire(main())