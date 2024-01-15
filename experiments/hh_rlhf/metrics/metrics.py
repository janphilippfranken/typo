import json
import logging

import fire
import hydra
from tqdm import tqdm
from omegaconf import DictConfig
from datasets import load_from_disk, load_dataset

from helpers import *

from scaituning.models.vllm_models.inference_model import VLLMInferenceModel


logging.basicConfig(level=logging.INFO)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(args: DictConfig) -> None:
    logging.info(f"Evaluating {args.constitution_file} using {args.model.name} on {args.split}")
    
   
    # get inference model
    is_vllm = "vllm" in args.model.model_type.lower()
    is_openai = "openai" in args.model.model_type.lower()
    
    if is_vllm:
        model = VLLMInferenceModel(**args.model.model_config)
      
    # get data
    data = load_dataset(**args.data.dataset)
    dataset = data[args.data.split] # test split 
    
    # get constitutions for evaluation
    constitutions = load_from_disk(f"{args.constitution_path}/{args.constitution_file}")
    
    final_constitutions = [[
        remove_numbering(constitution)
        for batch in constitutions['constitutions']
        for constitution in batch
    ][-1]]
    
    # get train examples
    train_examples = [
        batch['train_examples']
        for batch in constitutions
    ]
        
    # results dict
    results = {
        k: {} 
        for k, _ in enumerate(final_constitutions)
    }

    # main loop
    if "log_probs" in args.evaluation_prompt:
      
        if args.split == "train":
            examples = train_examples.copy()
            for batch_idx, constitution in tqdm(enumerate(final_constitutions)): # batch dimenstion
                for example_idx in tqdm(examples[batch_idx]): # train example dimension
                    log_prob_chosen, log_probs_rejected = run_eval_log_probs(
                        dataset=dataset,
                        constitution=constitution,
                        model=model,
                        eval_prompt=args.evaluation_prompt,
                        example_idx=example_idx,
                    )
                    results[batch_idx][example_idx] = {
                        'chosen': log_prob_chosen,
                        'rejected': log_probs_rejected,
                    }
        
        elif args.split == "test":
            examples = range(args.n_examples)
            for batch_idx, constitution in tqdm(enumerate(final_constitutions)): # batch dimenstion
                for example_idx in tqdm(examples): # train example dimension
                    log_prob_chosen, log_probs_rejected = run_eval_log_probs(
                        dataset=dataset,
                        constitution=constitution,
                        model=model,
                        eval_prompt=args.evaluation_prompt,
                        example_idx=example_idx,
                    )
                    results[batch_idx][example_idx] = {
                        'chosen': log_prob_chosen,
                        'rejected': log_probs_rejected,
                    }
                            
        # write to json
        with open(f"{args.storage_path}/{args.constitution_file}_model_{args.model.name}_{args.split}_n_final_{args.final_n}.json", "w") as f:
            json.dump(results, f)
    else:
        raise NotImplementedError


if __name__ == '__main__':
    fire.Fire(main())