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

    constitution_positive = CONSTITUTIONS[str(args.constitution_key)]["positive"]
    constitution_negative = CONSTITUTIONS[str(args.constitution_key)]["negative"][1]

    log_probs_positive_helpful = []
    log_probs_negative_helpful = []
    log_probs_positive_harmless = []
    log_probs_negative_harmless = []

    for constitution, question_helpful, question_harmless, response_helpful, response_harmless in tqdm(zip(
            data["constitution"],
            data["question_helpful"],
            data["question_harmless"],
            data["response_helpful"],
            data["response_harmless"]
    )):
        
        messages_helpful = [
            {"role": "user", "content": f"Write a response to the request below that follows the principles in the constitution.\n\nAssistant Constitution:\n{data['constitution'][constitution]}\n\n{data['question_helpful'][question_helpful]}"},
        ]
        encoded_helpful = model.tokenizer.decode(model.tokenizer.apply_chat_template(messages_helpful))
        prompt_helpful = encoded_helpful.split("<s>")[1].strip()
        response_formatted_helpful = f"{prompt_helpful}{data['response_helpful'][response_helpful]}"
        # breakpoint()
        logprob_pos_helpful = model.batch_log_probs([prompt_helpful], [response_formatted_helpful])
        
        messages_helpful_negative = [
            {"role": "user", "content": f"Write a response to the request below that follows the principles in the constitution.\n\nAssistant Constitution:\n{constitution_negative}\n\n{data['question_helpful'][question_helpful]}"},
        ]
        encoded_helpful_negative = model.tokenizer.decode(model.tokenizer.apply_chat_template(messages_helpful_negative))
        prompt_negative_helpful = encoded_helpful_negative.split("<s>")[1].strip()
        response_formatted_helpful = f"{prompt_negative_helpful}{data['response_helpful'][response_helpful]}"
        # breakpoint()
        logprob_neg_helpful = model.batch_log_probs([prompt_negative_helpful], [response_formatted_helpful])        
        
        log_probs_positive_helpful.append(logprob_pos_helpful.item())
        log_probs_negative_helpful.append(logprob_neg_helpful.item())

        
        # Harmless
        messages_harmless = [
            {"role": "user", "content": f"Write a response to the request below that follows the principles in the constitution.\n\nAssistant Constitution:\n{data['constitution'][constitution]}\n\n{data['question_harmless'][question_harmless]}"},
        ]
        encoded_harmless = model.tokenizer.decode(model.tokenizer.apply_chat_template(messages_harmless))
        prompt_harmless = encoded_harmless.split("<s>")[1].strip()
        response_formatted_harmless = f"{prompt_harmless}{data['response_harmless'][response_harmless]}"
        # breakpoint()
        logprob_pos_harmless = model.batch_log_probs([prompt_harmless], [response_formatted_harmless])
        
        messages_harmless_negative = [
            {"role": "user", "content": f"Write a response to the request below that follows the principles in the constitution.\n\nAssistant Constitution:\n{constitution_negative}\n\n{data['question_harmless'][question_harmless]}"},
        ]
        encoded_harmless_negative = model.tokenizer.decode(model.tokenizer.apply_chat_template(messages_harmless_negative))
        prompt_negative_harmless = encoded_harmless_negative.split("<s>")[1].strip()
        response_formatted_harmless = f"{prompt_negative_harmless}{data['response_harmless'][response_harmless]}"
        # breakpoint()
        logprob_neg_harmless = model.batch_log_probs([prompt_negative_harmless], [response_formatted_harmless])        
        
        log_probs_positive_harmless.append(logprob_pos_harmless.item())
        log_probs_negative_harmless.append(logprob_neg_harmless.item())

    log_probs_positive_negative_helpful = []
    log_probs_negative_negative_helpful = []
    log_probs_positive_negative_harmless = []
    log_probs_negative_negative_harmless = []

    for constitution, question_helpful, question_harmless, response_helpful, response_harmless in tqdm(zip(
            data_negative["constitution"],
            data_negative["question_helpful"],
            data_negative["question_harmless"],
            data_negative["response_helpful"],
            data_negative["response_harmless"]
    )):
        # Helpful
        messages_helpful = [
            {"role": "user", "content": f"Write a response to the request below that follows the principles in the constitution.\n\nAssistant Constitution:\n{data_negative['constitution'][constitution]}\n\n{data_negative['question_helpful'][question_helpful]}"},
        ]
        encoded_helpful = model.tokenizer.decode(model.tokenizer.apply_chat_template(messages_helpful))
        prompt_helpful = encoded_helpful.split("<s>")[1].strip()
        response_formatted_helpful = f"{prompt_helpful}{data_negative['response_helpful'][response_helpful]}"
        # breakpoint()
        logprob_pos_helpful = model.batch_log_probs([prompt_helpful], [response_formatted_helpful])
        
        messages_helpful_negative = [
            {"role": "user", "content": f"Write a response to the request below that follows the principles in the constitution.\n\nAssistant Constitution:\n{constitution_positive}\n\n{data_negative['question_helpful'][question_helpful]}"},
        ]
        encoded_helpful_negative = model.tokenizer.decode(model.tokenizer.apply_chat_template(messages_helpful_negative))
        prompt_negative_helpful = encoded_helpful_negative.split("<s>")[1].strip()
        response_formatted_helpful = f"{prompt_negative_helpful}{data_negative['response_helpful'][response_helpful]}"
        # breakpoint()
        logprob_neg_helpful = model.batch_log_probs([prompt_negative_helpful], [response_formatted_helpful])        
        
        log_probs_positive_negative_helpful.append(logprob_pos_helpful.item())
        log_probs_negative_negative_helpful.append(logprob_neg_helpful.item())
        
        # Harmless
        messages_harmless = [
            {"role": "user", "content": f"Write a response to the request below that follows the principles in the constitution.\n\nAssistant Constitution:\n{data_negative['constitution'][constitution]}\n\n{data_negative['question_harmless'][question_harmless]}"},
        ]
        encoded_harmless = model.tokenizer.decode(model.tokenizer.apply_chat_template(messages_harmless))
        prompt_harmless = encoded_harmless.split("<s>")[1].strip()
        response_formatted_harmless = f"{prompt_harmless}{data_negative['response_harmless'][response_harmless]}"
        # breakpoint()
        logprob_pos_harmless = model.batch_log_probs([prompt_harmless], [response_formatted_harmless])
        
        messages_harmless_negative = [
            {"role": "user", "content": f"Write a response to the request below that follows the principles in the constitution.\n\nAssistant Constitution:\n{constitution_positive}\n\n{data_negative['question_harmless'][question_harmless]}"},
        ]
        encoded_harmless_negative = model.tokenizer.decode(model.tokenizer.apply_chat_template(messages_harmless_negative))
        prompt_negative_harmless = encoded_harmless_negative.split("<s>")[1].strip()
        response_formatted_harmless = f"{prompt_negative_harmless}{data_negative['response_harmless'][response_harmless]}"
        # breakpoint()
        logprob_neg_harmless = model.batch_log_probs([prompt_negative_harmless], [response_formatted_harmless])   

        log_probs_positive_negative_harmless.append(logprob_pos_harmless.item())
        log_probs_negative_negative_harmless.append(logprob_neg_harmless.item())
        
    mut_inf_1_helpful = [r1 - r2 for r1, r2 in zip(log_probs_positive_helpful, log_probs_negative_helpful)]
    mut_inf_2_helpful = [r2 - r1 for r1, r2 in zip(log_probs_positive_negative_helpful, log_probs_negative_negative_helpful)]
    mut_inf_helpful = [r1 - r2 for r1, r2 in zip(mut_inf_1_helpful, mut_inf_2_helpful)]

    mut_inf_1_harmless = [r1 - r2 for r1, r2 in zip(log_probs_positive_harmless, log_probs_negative_harmless)]
    mut_inf_2_harmless = [r2 - r1 for r1, r2 in zip(log_probs_positive_negative_harmless, log_probs_negative_negative_harmless)]
    mut_inf_harmless = [r1 - r2 for r1, r2 in zip(mut_inf_1_harmless, mut_inf_2_harmless)]
    
    
    with open(f"{args.output_dir}/{args.file_name}-mutual-information-helpful.json", "w") as file:
        json.dump((mut_inf_helpful, np.mean(mut_inf_helpful), np.std(mut_inf_helpful) / np.sqrt(len(mut_inf_helpful))), file, indent=4)

    with open(f"{args.output_dir}/{args.file_name}-mutual-information-harmless.json", "w") as file:
        json.dump((mut_inf_harmless, np.mean(mut_inf_harmless), np.std(mut_inf_harmless) / np.sqrt(len(mut_inf_harmless))), file, indent=4)
        

if __name__ == "__main__":
    fire.Fire(main())