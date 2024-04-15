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
@hydra.main(version_base=None, config_path="conf", config_name="mutual_information_instruct")
def main(args: DictConfig) -> None:
    with open(f"{args.source_dir_positive}/{args.source_file_positive}", "r") as file:
        data_positive = json.load(file)

   
    with open(f"{args.source_dir_negative_both}/{args.source_file_negative_both}", "r") as file:
        data_negative_both = json.load(file)
        
    with open(f"{args.source_dir_negative_0}/{args.source_file_negative_0}", "r") as file:
        data_negative_0 = json.load(file)
        
    with open(f"{args.source_dir_negative_1}/{args.source_file_negative_1}", "r") as file:
        data_negative_1 = json.load(file)

    # model
    model = VLLMInferenceModel(**args.model_config)

    constitution = json.load(open(f"{args.constitution_dir}/2.json"))
    constitution_positive = constitution['positive']
    constitution_negative_both = constitution['negative']
    
    constitution = json.load(open(f"{args.constitution_dir}/0.json"))
    constitution_negative_0 = constitution['negative']
    
    constitution = json.load(open(f"{args.constitution_dir}/1.json"))
    constitution_negative_1 = constitution['negative']
    
    constitutions = [constitution_positive, constitution_negative_both, constitution_negative_0, constitution_negative_1]
    data = [data_positive, data_negative_both, data_negative_0, data_negative_1]

    

    mi_positive = None
    mi_negative_both = None
    mi_negative_0 = None
    mi_negative_1 = None
    
    breakpoint()
    
    def get_mi(data, constitutions, constitution_index):
        

        
        target_probs = []
        
        for question, response in tqdm(zip(data[constitution_index]["question"], data[constitution_index]["response"])): # loop over the data     
            messages = [
                {"role": "user", "content": f"Summarize the post below according to the principles in the constitution.\n\nSummarization Constitution:\n{shuffle_principles(constitutions[constitution_index])}\n\n{data[constitution_index]['question'][question]}\n\nSummary:"},
            ]
            encoded = model.tokenizer.decode(model.tokenizer.apply_chat_template(messages))
            prompt = encoded.split("<s>")[1].strip()
            response_formatted = f"{prompt}{data[constitution_index]['response'][response]}"
            logprob = model.batch_log_probs([prompt], [response_formatted])
            target_probs.append(logprob.item())
            

        other_probs = np.zeros((len(constitutions), len(data[constitution_index]["question"])))
        
        for i, constitution in enumerate(constitutions):
            
            for question, response in tqdm(zip(data[constitution_index]["question"], data[constitution_index]["response"])): # loop over the data     
                messages = [
                    {"role": "user", "content": f"Summarize the post below according to the principles in the constitution.\n\nSummarization Constitution:\n{shuffle_principles(constitutions[i])}\n\n{data[constitution_index]['question'][question]}\n\nSummary:"},
                ]
                encoded = model.tokenizer.decode(model.tokenizer.apply_chat_template(messages))
                prompt = encoded.split("<s>")[1].strip()
                response_formatted = f"{prompt}{data[constitution_index]['response'][response]}"
                logprob = model.batch_log_probs([prompt], [response_formatted])
                other_probs[i, int(question)] = logprob.item()
            
        other_probs = other_probs.mean(axis=0)
        mi = np.array(target_probs) - other_probs 
        
        return mi

        
    mi_positive = get_mi(data, constitutions, 0)
    # breakpoint()
    mi_negative_both = get_mi(data, constitutions, 1)
    mi_negative_0 = get_mi(data, constitutions, 2)
    mi_negative_1 = get_mi(data, constitutions, 3)
    
    mi_average = (mi_positive + mi_negative_both + mi_negative_0 + mi_negative_1) / 4
    breakpoint()
       
    with open(f"{args.output_dir}/{args.file_name}-mutual-information.json", "w") as file:
        json.dump(list(mi_average), file, indent=4)
    
    
if __name__ == "__main__":
    fire.Fire(main())
    
 