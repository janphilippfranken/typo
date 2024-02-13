import json
import logging

import fire
import hydra
from tqdm import tqdm
import numpy as np
from omegaconf import DictConfig
from datasets import load_from_disk, load_dataset
import torch
from helpers import *

from scaituning.models.vllm_models.inference_model import VLLMInferenceModel
from scaituning.models.huggingface_models.inference_model import HFInferenceModel


logging.basicConfig(level=logging.INFO)


@hydra.main(version_base=None, config_path="conf", config_name="metrics")
def main(args: DictConfig) -> None:
    logging.info(f"Evaluating {args.constitution_file} using {args.model.name} on {args.split}")
    
    # get inference model
    model = VLLMInferenceModel(**args.model.model_config)
   
    
    # get test data
    data = load_dataset(**args.data.dataset)
    dataset = data[args.data.split]
    breakpoint()
    
    # get constitutions for evaluation
    # constitutions = load_from_disk(f"{args.constitution_path}/{args.constitution_file}")
    
    # final_constitutions = [[
    #     remove_numbering(constitution)
    #     for batch in constitutions['constitutions']
    #     for constitution in batch
    # ][-1]]
    
    # final_constitution_strings = []
    # for constitution in final_constitutions:
    #     const_str = ""
    #     lines = constitution.split("\n")
    #     for i, line in enumerate(lines):
    #         const_str += f"{i + 1}. {line}\n"
    #     final_constitution_strings.append(const_str.strip())
    final_constitution_strings = ["The assistant should be helpful and not harmless."]
    final_constitutions = ["The assistant should be helpful and not harmless."]
    
 
    # # get train examples
    # train_examples = [
    #     batch['train_examples']
    #     for batch in constitutions
    # ]
        
    # results dict
    results = {
        k: {
            'train_labels': [],
            'train_logprobs': [],
            'train_probs': [],
            
            'test_labels': [],
            'test_logprobs': [],
            'test_probs': [],
            
        } 
        for k, _ in enumerate(final_constitutions)
    }
    
    logging.info("RUNNING ANSWER")
    # breakpoint()

    
    if args.split == "test":
        examples = range(args.start_of_test_examples, args.start_of_test_examples + args.n_examples)
        logging.info(examples)
        for batch_idx, constitution in tqdm(enumerate(final_constitution_strings)): #

            for example_idx in tqdm(examples):
                # breakpoint()
                print("reversed constitution")
                
                log_prob_chosen_constitution, log_prob_rejected_constitution, final_answer_chosen, final_answer_rejected = \
                    run_eval_answer(
                    dataset=dataset,
                    constitutions=[constitution],
                    model=model,
                    eval_prompt=args.evaluation_prompt,
                    example_idx=example_idx,
                )
                
                log_probs = np.array(
                    [
                        float(log_prob_chosen_constitution[batch_idx]), 
                        float(log_prob_rejected_constitution[batch_idx]),
                    ]
                )
                log_probs_average = np.array(
                    [
                        log_probs[0] / model.tokenizer(final_answer_chosen, return_tensors="pt").input_ids.shape[1],
                        log_probs[0] / model.tokenizer(final_answer_rejected, return_tensors="pt").input_ids.shape[1],
                    ]
                )
                
                
                probs = np.exp(log_probs) / np.sum(np.exp(log_probs))
                average_probs = np.exp(log_probs_average) / np.sum(np.exp(log_probs_average))
                
                label = np.argmax(probs)
                average_label = np.argmax(average_probs)
    
                results[batch_idx]['train_labels'].append(int(label))
                results[batch_idx]['train_logprobs'].append(list(log_probs))
                results[batch_idx]['train_probs'].append(probs[0])
                
                # Save progress after 
        with open(f"predictions/predictions_cdpo_not_harmless_epoch_1.json", "w") as f:
            json.dump(results, f)

if __name__ == '__main__':
    fire.Fire(main())
