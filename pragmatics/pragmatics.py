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


from prompts import CONSTITUTIONS


logging.basicConfig(level=logging.INFO)


@hydra.main(version_base=None, config_path="conf", config_name="pragmatics")
def main(args: DictConfig) -> None:
    
    # inference model 
    model = VLLMInferenceModel(**args.model.model_config)
    
    # dataset 
    data = load_dataset(**args.data.dataset)
    dataset = data[args.data.split]
 
    # loop over examples 
    for example_idx in range(args.n_examples):
        
        # response batch of shape: [constitution_batch_size, num_return_sequences]
        responses = \ 
            run_gen_answer(
            model=model,
            dataset=dataset,
            constitutions=CONSTITUTIONS, # sample these 
            eval_prompt=args.evaluation_prompt,
            example_idx=example_idx,
        )
            
        # compute logprobs of batch 
        logits = None 
        
        # normalize twice to get targets 
        targets = None 
        
        # compute cross entropy loss on class probabilities and logits 
        
        
                
 
if __name__ == '__main__':
    fire.Fire(main())
