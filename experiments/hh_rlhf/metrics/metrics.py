import logging

import fire
import hydra
import json
import torch
import pandas as pd
from tqdm import tqdm
from omegaconf import DictConfig

from datasets import load_dataset

from helpers import *


from scaituning.models.vllm_models.inference_model import VLLMInferenceModel

from prompts import EVALUATION_PROMPTS, SYSTEM_PROMPTS


logging.basicConfig(level=logging.INFO)


BOS_TOKEN, EOS_TOKEN = "<s>", "</s>"
B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(args: DictConfig) -> None:
    logging.info(f"Evaluating {args.metrics.constitution_file} using {args.model.name} on {args.metrics.split}")
    
   
    # GET INFERENCE MODEL
    is_vllm = "vllm" in args.model.model_type.lower()
    is_hf = "huggingface" in args.model.model_type.lower()
    is_openai = "openai" in args.model.model_type.lower()
    
    # if is_vllm:
        # model = VLLMInferenceModel(**args.model.model_config)
      
      
    # GET DATA
    data = load_dataset(**args.data.dataset)
    dataset = data[args.data.split] # test split 
    
    
    # GET CONSTITUTIONS FOR EVAL
    constitutions = load_from_disk(f"{args.metrics.constitution_path}/{args.metrics.constitution_file}")
    
    
    # if len(constitutions) == 1:
    final_constitutions = [[
        remove_numbering(constitution)
        for batch in constitutions['constitutions']
        for constitution in batch
    ][-1]]
    
    breakpoint()

    
    train_examples = [
        batch['train_examples']
        for batch in constitutions
    ]
        
    
    # RESULTS DICT
    results = {
        k: {} 
        for k, _ in enumerate(final_constitutions)
    }

    
    # MAIN LOOP 
    if "log_probs" in args.metrics.evaluation_prompt:
        
        
                    
        if args.metrics.split == "train":
            examples = train_examples.copy()
            for batch_idx, constitution in tqdm(enumerate(final_constitutions)): # batch dimenstion
                for example_idx in tqdm(examples[batch_idx]): # train example dimension
                    log_prob_chosen, log_probs_rejected = run_eval_log_probs(
                        dataset=dataset,
                        constitution=constitution,
                        model=model,
                        eval_prompt=args.metrics.evaluation_prompt,
                        example_idx=example_idx,
                    )
                    results[batch_idx][example_idx] = {
                        'chosen': log_prob_chosen,
                        'rejected': log_probs_rejected,
                    }
        
        elif args.metrics.split == "test":
            examples = range(args.metrics.n_examples)
            for batch_idx, constitution in tqdm(enumerate(final_constitutions)): # batch dimenstion
                for example_idx in tqdm(examples): # train example dimension
                    log_prob_chosen, log_probs_rejected = run_eval_log_probs(
                        dataset=dataset,
                        constitution=constitution,
                        model=model,
                        eval_prompt=args.metrics.evaluation_prompt,
                        example_idx=example_idx,
                    )
                    results[batch_idx][example_idx] = {
                        'chosen': log_prob_chosen,
                        'rejected': log_probs_rejected,
                    }
                            
    
        # WRITE TO JSON
        with open(f"{args.metrics.storage_path}/{args.metrics.constitution_file}_model_{args.model.name}_{args.metrics.split}_n_final_{args.metrics.final_n}.json", "w") as f:
            json.dump(results, f)

        
if __name__ == '__main__':
    fire.Fire(main())