import json
import fire
import hydra
import numpy as np
from tqdm import tqdm
import random
from omegaconf import DictConfig
from datasets import load_dataset
import torch
from typo.models.vllm_models.inference_model import VLLMInferenceModel
from helpers import *
from prompts import *

np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)

# main evaluation script
@hydra.main(version_base=None, config_path="conf", config_name="mutual_information_clean_llama")
def main(args: DictConfig) -> None:
    # data
    with open(f"{args.source_dir_positive}/{args.source_file_positive}", "r") as file:
        data_positive = json.load(file)
   
    with open(f"{args.source_dir_negative}/{args.source_file_negative}", "r") as file:
        data_negative = json.load(file)
        

    # model
    model = VLLMInferenceModel(**args.model_config)
    
    data = [data_positive, data_negative]

    

    mi_positive = None
    mi_negative = None
    
 
    
    def get_mi(data, constitution_index):
        
        target_probs = []

        for i, (question, response) in tqdm(enumerate(zip(data[constitution_index]["question"], data[constitution_index]["response"]))): # loop over the data     
            prompt = PROMPT_TRAINING.format(constitution=data[constitution_index]["constitution"][str(i)], question=data[constitution_index]["question"][question])
            response_formatted = f"{prompt}{data[constitution_index]['response'][response]}"
            logprob = model.batch_log_probs([prompt], [response_formatted])
            target_probs.append(logprob.item())
            
            
        other_probs = np.zeros((len(data), len(data[constitution_index]["question"])))
        
        for c_idx in [0, 1]:
            
            for i, (question, response) in tqdm(enumerate(zip(data[constitution_index]["question"], data[constitution_index]["response"]))): # loop over the data     
                prompt = PROMPT_TRAINING.format(constitution=data[c_idx]["constitution"][str(i)], question=data[constitution_index]["question"][question])
                response_formatted = f"{prompt}{data[constitution_index]['response'][response]}"
                logprob = model.batch_log_probs([prompt], [response_formatted])
                other_probs[c_idx, int(question)] = logprob.item()
            
            
        other_probs = other_probs.mean(axis=0)
        mi = np.array(target_probs) - other_probs 
        
        return mi

        
    mi_positive = get_mi(data, 0)
    mi_negative = get_mi(data, 1)
   
    mi_average = (mi_positive + mi_negative) / 2
    breakpoint()
       
    with open(f"{'results_fixed/mutual_information_llama'}/{args.file_name}-mutual-information.json", "w") as file:
        json.dump(list(mi_average), file, indent=4)

if __name__ == "__main__":
    fire.Fire(main())
    
 