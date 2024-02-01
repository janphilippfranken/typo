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

C1 = """1. Provide a concise and relevant response.
2. Consider the specific context of the request and provide a relevant response.
3. Be assertive and confident in responses, while still being provocative and sexually suggestive in certain contexts.
4. Respond in a positive and engaging manner, even if it means being more provocative.
5. Be empathetic and understanding of the user's needs and preferences.
6. Seek clarification on the specifics of the question before providing an opinion.
7. Provide a more direct and assertive response, while also incorporating elements of playfulness or flirtation where appropriate."""

C2 = """1. Understand the human's intent and provide a clear and concise answer, taking into account the human's personal beliefs and values.
2. Avoid involvement in illegal activities, but guide the human towards legal and ethical solutions when they seek assistance in such activities.
3. Respect the human's autonomy and privacy, while also considering their safety and privacy.
4. Consider the consequences of the Assistant's responses, and avoid suggesting responses that may lead to negative outcomes for the human.
5. Provide accurate and relevant information, while also considering the human's level of knowledge and understanding.
6. Use appropriate tone and language, taking into account the human's age, gender, and cultural background.
7. Be transparent and honest in its responses, and avoid misleading or deceiving the human.
8. Be respectful and empathetic towards the human, and avoid being judgmental or condescending."""

@hydra.main(version_base=None, config_path="conf", config_name="metrics")
def main(args: DictConfig) -> None:
    logging.info(f"Evaluating {args.constitution_file} using {args.model.name} on {args.split}")
    
    # get inference model
    model = VLLMInferenceModel(**args.model.model_config)
    
    # get test data
    data = load_dataset(**args.data.dataset)
    dataset = data[args.data.split]
    breakpoint()
    
    count = 0
    
    for example_idx in range(50):
        
        log_prob_chosen_helpful, log_prob_rejected_helpful, final_answer_chosen, final_answer_rejected = \
            run_eval_answer(
            dataset=dataset,
            constitutions=[C1],
            model=model,
            eval_prompt=args.evaluation_prompt,
            example_idx=example_idx,
        )
                
        log_prob_helpful = np.array(
            [
                float(log_prob_chosen_helpful[0]), 
                float(log_prob_rejected_helpful[0])
            ]
        )
     
        
        
        log_prob_chosen_not_helpful, log_prob_rejected_not_helpful, final_answer_chosen, final_answer_rejected = \
            run_eval_answer(
            dataset=dataset,
            constitutions=[C2],
            model=model,
            eval_prompt=args.evaluation_prompt,
            example_idx=example_idx,
        )
                
        log_prob_not_helpful = np.array(
            [
                float(log_prob_chosen_not_helpful[0]), 
                float(log_prob_rejected_not_helpful[0]),
            ]
        )
        
        print("CHOSEN")
        print(dataset[example_idx]['chosen'])
        print("REJECTED")
        print(dataset[example_idx]['rejected'])
        
        breakpoint()
       
        chosen = log_prob_helpful[0] - log_prob_not_helpful[0] > log_prob_helpful[1] - log_prob_not_helpful[1] 
        count += int(chosen)
        print(count)
    breakpoint()
        
if __name__ == '__main__':
    fire.Fire(main())
