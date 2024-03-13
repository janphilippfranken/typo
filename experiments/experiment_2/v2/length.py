import json
import fire
import hydra
import random
import numpy as np
from omegaconf import DictConfig
from tqdm import tqdm


from typo.models.openai_models.gpt4 import GPT4Agent
from typo.models.openai_models.azure import AsyncAzureChatLLM

from transformers import AutoTokenizer, AutoModelForCausalLM


from prompts import GPT4_WIN_RATE, SYSTEM_MESSAGE


def load_model_responses(filename):
    with open(filename, "r") as file:
        return json.load(file)
    

@hydra.main(version_base=None, config_path="conf", config_name="length")
def main(args: DictConfig) -> None:
    
    model_baseline = load_model_responses(f"{args.model_dir}/{args.baseline}.json")
    model_test = load_model_responses(f"{args.model_dir}/{args.test}.json")
    
    # get tokenizer    
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path="mistralai/Mistral-7B-v0.1",
        cache_dir="/scr/jphilipp/typo/pretrained_models/Mistral-7B-v0.1",
        model_max_length=2048,
    )

    
    length_base_helpful = []
    length_test_helpful = []
    
    length_base_harmless = []
    length_test_harmless = []
    
    constitutions = list(model_baseline['constitution'].values())

    print(len(constitutions))
    
    for i, _ in enumerate(constitutions):

        
        length_base_helpful.append(
            len(tokenizer.encode(model_baseline['response_helpful'][str(i)].split("\n\n3.")[0].strip()))
        )
        
        length_test_helpful.append(
            len(tokenizer.encode(model_test['response_helpful'][str(i)].split("\n\n3.")[0].strip()))
        )
        
        length_base_harmless.append(
            len(tokenizer.encode(model_baseline['response_harmless'][str(i)].split("\n\n3.")[0].strip()))
        )
        
        length_test_harmless.append(
            len(tokenizer.encode(model_test['response_harmless'][str(i)].split("\n\n3.")[0].strip()))
        )
        
        
        with open(f'{args.output_dir}/{args.helpful_win_rates_file_name}_length_base_helpful.json', 'w') as file:
            json.dump(length_base_helpful, file, indent=4)
            
        with open(f'{args.output_dir}/{args.helpful_win_rates_file_name}_length_test_helpful.json', 'w') as file:
            json.dump(length_test_helpful, file, indent=4)
            
        with open(f'{args.output_dir}/{args.helpful_win_rates_file_name}_length_base_harmless.json', 'w') as file:
            json.dump(length_base_harmless, file, indent=4)
            
        with open(f'{args.output_dir}/{args.helpful_win_rates_file_name}_length_test_harmless.json', 'w') as file:
            json.dump(length_test_harmless, file, indent=4)
 

if __name__ == "__main__":
    fire.Fire(main())