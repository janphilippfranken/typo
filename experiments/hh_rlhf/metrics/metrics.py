import json
import logging

import fire
import hydra
from tqdm import tqdm
import numpy as np
from omegaconf import DictConfig
from datasets import load_from_disk, load_dataset

from helpers import *

from scaituning.models.vllm_models.inference_model import VLLMInferenceModel
from scaituning.models.huggingface_models.inference_model import HFInferenceModel


logging.basicConfig(level=logging.INFO)


@hydra.main(version_base=None, config_path="conf", config_name="metrics")
def main(args: DictConfig) -> None:
    logging.info(f"Evaluating {args.constitution_file} using {args.model.name} on {args.split}")
    
    # get inference model
    is_vllm = "vllm" in args.model.model_type.lower()
    is_hf = "huggingface" in args.model.model_type.lower()
    
    if is_vllm:
        model = VLLMInferenceModel(**args.model.model_config)
    elif is_hf:
        model = HFInferenceModel(**args.model.model_config)
        model.model.is_parallelizable = True
        model.model.model_parallel = True
              
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
        k: {
            'train_labels': [],
            'train_logprobs': [],
            'train_probs': [],
            'train_labels_average': [],
            'train_logprobs_average': [],
            'train_probs_average': [],
            'train_response': [],
            
            'test_labels': [],
            'test_logprobs': [],
            'test_probs': [],
        } 
        for k, _ in enumerate(final_constitutions)
    }
    
    
        
    elif "answer" in args.evaluation_prompt:
        logging.info("RUNNING ANSWER")
    
        if args.split == "train":
            examples = train_examples.copy()
            for batch_idx, constitution in tqdm(enumerate(final_constitutions)): #
                for example_idx in tqdm(examples[batch_idx]): 
            
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
                    
                    results[batch_idx]['train_labels_average'].append(int(average_label))
                    results[batch_idx]['train_logprobs_average'].append(list(log_probs_average))
                    results[batch_idx]['train_probs_average'].append(average_probs[0])
                    
       
        
            # Save progress after 
            with open(f"{args.storage_path}/{args.constitution_file}_model_{args.model.name}_{args.split}_answer.json", "w") as f:
                json.dump(results, f)
        
        elif args.split == "test":
            examples = dataset[args.start_of_test_examples:args.start_of_test_examples + args.n_examples]
            for batch_idx, constitution in tqdm(enumerate(final_constitutions)): #
                for example_idx in tqdm(examples[batch_idx]): 
            
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
                    
                    results[batch_idx]['train_labels_average'].append(int(average_label))
                    results[batch_idx]['train_logprobs_average'].append(list(log_probs_average))
                    results[batch_idx]['train_probs_average'].append(average_probs[0])
                    
                 # Save progress after 
            with open(f"{args.storage_path}/{args.constitution_file}_model_{args.model.name}_{args.split}_answer.json", "w") as f:
                json.dump(results, f)
    
if __name__ == '__main__':
    fire.Fire(main())