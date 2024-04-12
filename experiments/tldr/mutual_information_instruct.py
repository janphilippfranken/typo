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
    # data
    with open(f"{args.source_dir_positive}/{args.source_file_positive}", "r") as file:
        data = json.load(file)

    # data
    with open(f"{args.source_dir_negative}/{args.source_file_negative}", "r") as file:
        data_negative = json.load(file)

    # model
    model = VLLMInferenceModel(**args.model_config)

    constitution = json.load(open(f"{args.constitution_dir}/2.json"))
    constitution_negative = constitution['negative']
    constitution_positive = constitution['positive']

    log_probs_positive = []
    log_probs_negative = []

    for constitution, question, response in tqdm(zip(data["constitution"], data["question"], data["response"])):
          
        messages = [
            {"role": "user", "content": f"Summarize the post below according to the principles in the constitution.\n\nSummarization Constitution:\n{data['constitution'][constitution]}\n\n{data['question'][question]}\n\nSummary:"},
        ]
        encoded = model.tokenizer.decode(model.tokenizer.apply_chat_template(messages))
        prompt = encoded.split("<s>")[1].strip()
        response_formatted = f"{prompt}{data['response'][response]}"

        logprob_pos = model.batch_log_probs([prompt], [response_formatted])
        
        messages_negative = [
            {"role": "user", "content": f"Summarize the post below according to the principles in the constitution.\n\nSummarization Constitution:\n{constitution_negative}\n\n{data['question'][question]}\n\nSummary:"},
        ]
        encoded = model.tokenizer.decode(model.tokenizer.apply_chat_template(messages_negative))
        prompt_negative = encoded.split("<s>")[1].strip()
        response_formatted = f"{prompt_negative}{data['response'][response]}"
        logprob_neg = model.batch_log_probs([prompt_negative], [response_formatted])
        

        log_probs_positive.append(logprob_pos.item())
        log_probs_negative.append(logprob_neg.item())

    log_probs_positive_negative = []
    log_probs_negative_negative = []

    for constitution, question, response in tqdm(zip(data_negative["constitution"], data_negative["question"], data_negative["response"])):
        messages = [
            {"role": "user", "content": f"Summarize the post below according to the principles in the constitution.\n\nSummarization Constitution:\n{data_negative['constitution'][constitution]}\n\n{data_negative['question'][question]}\n\nSummary:"},
        ]
        encoded = model.tokenizer.decode(model.tokenizer.apply_chat_template(messages))
        prompt = encoded.split("<s>")[1].strip()
        response_formatted = f"{prompt}{data_negative['response'][response]}"

        logprob_pos = model.batch_log_probs([prompt], [response_formatted])
        
        messages_negative = [
            {"role": "user", "content": f"Summarize the post below according to the principles in the constitution.\n\nSummarization Constitution:\n{constitution_positive}\n\n{data_negative['question'][question]}\n\nSummary:"},
        ]
        encoded = model.tokenizer.decode(model.tokenizer.apply_chat_template(messages_negative))
        prompt_negative = encoded.split("<s>")[1].strip()
        response_formatted = f"{prompt_negative}{data_negative['response'][response]}"
        logprob_neg = model.batch_log_probs([prompt_negative], [response_formatted])
        
        log_probs_positive_negative.append(logprob_pos.item())
        log_probs_negative_negative.append(logprob_neg.item())

    mut_inf_1 = [r1 - r2 for r1, r2 in zip(log_probs_positive, log_probs_negative)]
    mut_inf_2 = [r2 - r1 for r1, r2 in zip(log_probs_positive_negative, log_probs_negative_negative)]
    mut_inf = [r1 - r2 for r1, r2 in zip(mut_inf_1, mut_inf_2)]
    # this here is the same as above just needst o be divided by 2 instead of 4
    # mut_inf_1 = [r1 - np.mean([r1, r2]) for r1, r2 in zip(log_probs_positive, log_probs_negative)]
    # mut_inf_2 = [r2 - np.mean([r1, r2]) for r1, r2 in zip(log_probs_negative_negative, log_probs_positive_negative)]
    # mut_inf = [r1 + r2 for r1, r2 in zip(mut_inf_1, mut_inf_2)]
    breakpoint()

    with open(f"{args.output_dir}/{args.file_name}-mutual-information.json", "w") as file:
        json.dump((mut_inf, np.mean(mut_inf), np.std(mut_inf) / np.sqrt(len(mut_inf))), file, indent=4)

if __name__ == "__main__":
    fire.Fire(main())
    
 