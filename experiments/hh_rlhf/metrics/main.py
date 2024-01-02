import logging

import fire
import hydra
import torch
import pandas as pd
from tqdm import tqdm
from omegaconf import DictConfig

from datasets import load_dataset

from helpers import *

from scaituning.models.huggingface_models.inference_model import HFInferenceModel

from prompts import EVALUATION_PROMPTS


logging.basicConfig(level=logging.INFO)


@hydra.main(version_base=None, config_name="config")
def main(args: DictConfig) -> None:
    
   
    # GET INFERENCE MODEL
    model_generate = HFInferenceModel(**args.model_config)
    
    # GET DATA
    data = load_dataset(**args.dataset)
    dataset = data[args.split]
    
    # LOAD CONSTITUTIONS
    constitutions = pd.read_csv('constitutions_1.csv')
    constitutions = constitutions.drop(columns=['Unnamed: 0'])
    rlhf_1 = []
    for constitution in constitutions['rlhf_1']:
        rlhf_1.append(remove_numbering(constitution))
    rlhf_reversed_1 = []
    for constitution in constitutions['rlhf_reversed_1']:
        rlhf_reversed_1.append(remove_numbering(constitution))

        
    # MAIN LOOP
    for example_idx in tqdm(range(args.n_examples)):
        
        example_chosen = dataset[example_idx]['chosen']
        example_rejected = dataset[example_idx]['rejected']

        conversation_chosen, final_answer_chosen = remove_final_answer(example_chosen)
        conversation_rejected, final_answer_rejected = remove_final_answer(example_rejected)


        prompt = build_eval_prompt(
            constitution=rlhf_1[0],
            prompt_template=EVALUATION_PROMPTS['evaluation_prompt_1'],
            conversation=conversation_chosen,
            answer_chosen=final_answer_chosen,
            answer_rejected=final_answer_rejected,
        )
        
        # responses = model.batch_prompt(
        #     formatted_prompts,
        #     **args.model_generate.completion_config,
        # )
        breakpoint()
        
    
        
        
        

       
       
if __name__ == '__main__':
    fire.Fire(main())