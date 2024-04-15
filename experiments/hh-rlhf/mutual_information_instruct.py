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
@hydra.main(version_base=None, config_path="conf", config_name="mutual_information")
def main(args: DictConfig) -> None:
    # data
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
    
    

    constitution_helpful = CONSTITUTIONS['helpful']
    constitution_harmless = CONSTITUTIONS['harmless']
    
    constitution_both_positive = constitution_helpful['positive']
    constitution_both_negative = constitution_helpful['negative'][1]
    constitution_0_negative = constitution_helpful['negative'][0]
    constitution_1_negative = constitution_harmless['negative'][0]
    
    
    constitutions = [constitution_both_positive, constitution_both_negative, constitution_0_negative, constitution_1_negative]
    data = [data_positive, data_negative_both, data_negative_0, data_negative_1]    

    
    
    def get_mi(data, constitutions, constitution_index):
        
        target_probs_helpful = []
        target_probs_harmless = []
        

        for question_helpful, question_harmless, response_helpful, response_harmless in tqdm(zip(
            data[constitution_index]["question_helpful"],
            data[constitution_index]["question_harmless"],
            data[constitution_index]["response_helpful"],
            data[constitution_index]["response_harmless"]
        )):

            
            messages_helpful = [
                {"role": "user", "content": f"Write a response to the request below that follows the principles in the constitution.\n\nAssistant Constitution:\n{shuffle_principles(constitutions[constitution_index])}\n\n{data[constitution_index]['question_helpful'][question_helpful]}"},
            ]
            encoded_helpful = model.tokenizer.decode(model.tokenizer.apply_chat_template(messages_helpful))
            prompt_helpful = encoded_helpful.split("<s>")[1].strip()
            response_formatted_helpful = f"{prompt_helpful}{data[constitution_index]['response_helpful'][response_helpful]}"
            logprob_helpful = model.batch_log_probs([prompt_helpful], [response_formatted_helpful])
            target_probs_helpful.append(logprob_helpful.item())
            
            messages_harmless = [
                {"role": "user", "content": f"Write a response to the request below that follows the principles in the constitution.\n\nAssistant Constitution:\n{shuffle_principles(constitutions[constitution_index])}\n\n{data[constitution_index]['question_helpful'][question_helpful]}"},
            ]
            encoded_harmless = model.tokenizer.decode(model.tokenizer.apply_chat_template(messages_harmless))
            prompt_harmless = encoded_harmless.split("<s>")[1].strip()
            response_formatted_harmless = f"{prompt_harmless}{data[constitution_index]['response_harmless'][response_harmless]}"
            logprob_harmless = model.batch_log_probs([prompt_harmless], [response_formatted_harmless])
            target_probs_harmless.append(logprob_harmless.item())
            

        other_probs_helpful = np.zeros((len(constitutions), len(data[constitution_index]["question_helpful"])))
        other_probs_harmless = np.zeros((len(constitutions), len(data[constitution_index]["question_harmless"])))
        
        for i, constitution in enumerate(constitutions):
            
            for question_helpful, question_harmless, response_helpful, response_harmless in tqdm(zip(
                data[constitution_index]["question_helpful"],
                data[constitution_index]["question_harmless"],
                data[constitution_index]["response_helpful"],
                data[constitution_index]["response_harmless"]
            )):
            
                messages_helpful = [
                {"role": "user", "content": f"Write a response to the request below that follows the principles in the constitution.\n\nAssistant Constitution:\n{shuffle_principles(constitutions[constitution_index])}\n\n{data[constitution_index]['question_helpful'][question_helpful]}"},
                ]
                encoded_helpful = model.tokenizer.decode(model.tokenizer.apply_chat_template(messages_helpful))
                prompt_helpful = encoded_helpful.split("<s>")[1].strip()
                response_formatted_helpful = f"{prompt_helpful}{data[i]['response_helpful'][response_helpful]}"
                logprob_helpful = model.batch_log_probs([prompt_helpful], [response_formatted_helpful])
                other_probs_helpful[i, int(question_helpful)] = logprob_helpful.item()
                
                messages_harmless = [
                    {"role": "user", "content": f"Write a response to the request below that follows the principles in the constitution.\n\nAssistant Constitution:\n{shuffle_principles(constitutions[constitution_index])}\n\n{data[constitution_index]['question_helpful'][question_helpful]}"},
                ]
                encoded_harmless = model.tokenizer.decode(model.tokenizer.apply_chat_template(messages_harmless))
                prompt_harmless = encoded_harmless.split("<s>")[1].strip()
                response_formatted_harmless = f"{prompt_harmless}{data[i]['response_harmless'][response_harmless]}"
                logprob_harmless = model.batch_log_probs([prompt_harmless], [response_formatted_harmless])
                other_probs_harmless[i, int(question_harmless)] = logprob_harmless.item()
            
            
        other_probs_helpful = other_probs_helpful.mean(axis=0)
        other_probs_harmless = other_probs_harmless.mean(axis=0)
        mi_helpful = np.array(target_probs_helpful) - other_probs_helpful
        mi_harmless = np.array(target_probs_harmless) - other_probs_harmless
        
        return mi_helpful, mi_harmless

        
    mi_positive_helpful, mi_positive_harmless = get_mi(data, constitutions, 0)
    breakpoint()
    mi_negative_both_helpful, mi_negative_both_harmless = get_mi(data, constitutions, 1)
    mi_negative_0_helpful, mi_negative_0_harmless = get_mi(data, constitutions, 2)
    mi_negative_1_helpful, mi_negative_1_harmless = get_mi(data, constitutions, 3)
    
    mi_average_helpful = (mi_positive_helpful + mi_negative_both_helpful + mi_negative_0_helpful + mi_negative_1_helpful) / 4
    mi_average_harmless = (mi_positive_harmless + mi_negative_both_harmless + mi_negative_0_harmless + mi_negative_1_harmless) / 4
    breakpoint()
       
    with open(f"{args.output_dir}/{args.file_name}-mutual-information-helpful.json", "w") as file:
        json.dump(list(mi_average_helpful), file, indent=4)
        
    with open(f"{args.output_dir}/{args.file_name}-mutual-information-harmless.json", "w") as file:
        json.dump(list(mi_average_harmless), file, indent=4)

if __name__ == "__main__":
    fire.Fire(main())
    
    
    
    
    