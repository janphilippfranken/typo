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

def mutual_information(log_probs):
    
    sum_mi = 0
    for log_prob in log_probs:
        sum_mi += 1/len(log_probs) * (log_prob - np.mean(log_probs))
        
    return sum_mi

def get_log_probs(model, data, constitutions):

    log_probs = {}
    
    for i, constitution in enumerate(constitutions):
        logprobs = []
        for question, response in tqdm(zip(data["question"], data["response"])):
            prompt = PROMPT_TRAINING.format(constitution=constitution, question=data["question"][question])
            response_formatted = f"{prompt}{data['response'][response]}"
            logprob = model.batch_log_probs([prompt], [response_formatted])
            logprobs.append(logprob.item())
        log_probs[i] = logprobs
        
    return log_probs

# main evaluation script
@hydra.main(version_base=None, config_path="conf", config_name="mutual_information_clean")
def main(args: DictConfig) -> None:
   
    # model
    model = VLLMInferenceModel(**args.model_config)
    breakpoint()
    constitutions = []
    constitution_both = json.load(open(f"{args.constitution_dir}/2.json"))
    constitutions.append(constitution_both["positive"])
    constitutions.append(constitution_both["negative"])
    constitution_pos_concise = json.load(open(f"{args.constitution_dir}/1.json"))
    constitutions.append(constitution_pos_concise["negative"])
    constitution_neg_concise = json.load(open(f"{args.constitution_dir}/0.json"))
    constitutions.append(constitution_neg_concise["negative"])
    breakpoint()
    
    file_paths = [
        "base-mistral-mistral-constitution-temperature-0.0.json",
        "base-mistral-mistral-constitution-temperature-0.0-negative.json",
        "base-mistral-mistral-constitution-neg-concise-pos-comprehensive-temperature-0.0-negative"
        
    ]
    
    # with open(f"{args.output_dir}/{args.file_name}-mutual-information.json", "w") as file:
    #     json.dump((mut_inf, np.mean(mut_inf), np.std(mut_inf) / np.sqrt(len(mut_inf))), file, indent=4)

if __name__ == "__main__":
    fire.Fire(main())
    
 