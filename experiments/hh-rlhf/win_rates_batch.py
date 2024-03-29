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
    

@hydra.main(version_base=None, config_path="conf", config_name="win_rates")
def main(args: DictConfig) -> None:
    
    model_baseline = load_model_responses(f"{args.model_dir}/{args.baseline}")
    model_test = load_model_responses(f"{args.model_dir}/{args.test}")
    
    print("BASELINE", args.baseline)
    print("TEST", args.test)
    
    # get tokenizer    
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path="mistralai/Mistral-7B-Instruct-v0.1",
        cache_dir="/scr/jphilipp/typo/pretrained_models/Mistral-7B-Instruct-v0.1",
        model_max_length=2048,
    )
    
    win_rates_helpful = []
    win_rates_harmless = []
    length = []
    
    llm = AsyncAzureChatLLM(
        azure_endpoint="https://philipp.openai.azure.com/",
        api_version="2023-05-15",
    )
    
    model = GPT4Agent(
        llm=llm,
        model="gpt-4",
        temperature=0.0,
        top_p=0.9,
        max_tokens=200,
        n=1,
    )

    constitutions = list(model_baseline['constitution'].values())
    # breakpoint()
    questions_helpful = list(model_baseline['question_helpful'].values())
    question_harmless = list(model_baseline['question_harmless'].values())

    
    # print(len(constitutions))
    random.seed(1)
    np.random.seed(1)
    prompts_helpful = []
    prompts_harmless = []
    numbers = []
    lengths_helpful = []
    lengths_harmless = []
    
    for i, (constitution, question_helpful, question_harmless) in enumerate(zip(constitutions, questions_helpful, question_harmless)):
        print('running')
    
        if i >= 250: 
            continue
        
        principles = [principle.strip()[3:] for i, principle in enumerate(constitution.split("\n"))]
        random.shuffle(principles)
        principles = [f"{i+1}. " + principle for i, principle in enumerate(principles)]
        constitution_shuffled = "\n".join(principles) 
        length_helpful = len(tokenizer.encode(model_test['response_helpful'][str(i)]))
        lengths_helpful.append(length_helpful)
        length_harmless = len(tokenizer.encode(model_test['response_harmless'][str(i)]))
        lengths_harmless.append(length_harmless)

        rand_number = np.random.randint(2)
        numbers.append(rand_number)
        
        if rand_number == 0:
        
            prompt_helpful = GPT4_WIN_RATE.format(
                constitution=constitution_shuffled,
                query=question_helpful,
                response_a=model_baseline['response_helpful'][str(i)],
                response_b=model_test['response_helpful'][str(i)],
            )
            prompts_helpful.append(prompt_helpful)
            
            prompt_harmless = GPT4_WIN_RATE.format(
                constitution=constitution_shuffled,
                query=question_harmless,
                response_a=model_baseline['response_harmless'][str(i)],
                response_b=model_test['response_harmless'][str(i)],
            )
            prompts_harmless.append(prompt_harmless)
              
        
        elif rand_number == 1:
            prompt_helpful = GPT4_WIN_RATE.format(
                constitution=constitution_shuffled,
                query=question_helpful,
                response_b=model_baseline['response_helpful'][str(i)],
                response_a=model_test['response_helpful'][str(i)],
            )
               
            prompts_helpful.append(prompt_helpful)
            
            prompt_harmless = GPT4_WIN_RATE.format(

                constitution=constitution_shuffled,
                query=question_harmless,
                response_b=model_baseline['response_harmless'][str(i)],
                response_a=model_test['response_harmless'][str(i)],
            )
            
            prompts_harmless.append(prompt_harmless)
        
    # breakpoint()
    # breakpoint()
    responses_helpful = model.batch_prompt(
        system_message=SYSTEM_MESSAGE,
        messages=prompts_helpful,
        win_rates=True,
    )
    
    responses_harmless = model.batch_prompt(
        system_message=SYSTEM_MESSAGE,
        messages=prompts_harmless,
        win_rates=True,
    )
    # breakpoint()
    
    formatted_responses_helpful = []
    # breakpoint()
    for response in responses_helpful:
        try:
            formatted_responses_helpful.append(response[0].split('Final Response:')[1].strip())
        except:
            formatted_responses_helpful.append("C")

    for formatted_response, number, length in zip(formatted_responses_helpful, numbers, lengths_helpful):
        print(formatted_response, number)
        if number == 0:

            
            if 'A' in formatted_response and 'B' not in formatted_response:
                win_rates_helpful.append((0, length))
                
            elif 'A' in formatted_response and 'B' in formatted_response:
                win_rates_helpful.append((0.5, length))
                
            elif 'A' not in formatted_response and 'B' in formatted_response:
                win_rates_helpful.append((1, length))
            else:
                print("ERROR")
                win_rates_helpful.append((0.5, length))
        
        elif number == 1:
            if 'A' in formatted_response and 'B' not in formatted_response:
                win_rates_helpful.append((1, length))
                
            elif 'A' in formatted_response and 'B' in formatted_response:
                win_rates_helpful.append((0.5, length))
                
            elif 'A' not in formatted_response and 'B' in formatted_response:
                win_rates_helpful.append((0, length))
            else:
                print("ERROR")
                win_rates_helpful.append((0.5, length))
                

    with open(f'{args.output_dir}/{args.win_rates_file_name}-helpful.json', 'w') as file:
        json.dump(win_rates_helpful, file, indent=4)
        
        
    formatted_responses_harmless = []
    
    for response in responses_harmless:
        try:
            formatted_responses_harmless.append(response[0].split('Final Response:')[1].strip())
        except:
            formatted_responses_harmless.append("C")

    for formatted_response, number, length in zip(formatted_responses_harmless, numbers, lengths_harmless):
        print(formatted_response, number)
        if number == 0:
            if 'A' in formatted_response and 'B' not in formatted_response:
                win_rates_harmless.append((0, length))
                
            elif 'A' in formatted_response and 'B' in formatted_response:
                win_rates_harmless.append((0.5, length))
                
            elif 'A' not in formatted_response and 'B' in formatted_response:
                win_rates_harmless.append((1, length))
            else:
                print("ERROR")
                win_rates_harmless.append((0.5, length))
        
        elif number == 1:
            if 'A' in formatted_response and 'B' not in formatted_response:
                win_rates_harmless.append((1, length))
                
            elif 'A' in formatted_response and 'B' in formatted_response:
                win_rates_harmless.append((0.5, length))
                
            elif 'A' not in formatted_response and 'B' in formatted_response:
                win_rates_harmless.append((0, length))
            else:
                print("ERROR")
                win_rates_harmless.append((0.5, length))
                
    with open(f'{args.output_dir}/{args.win_rates_file_name}-harmless.json', 'w') as file:
        json.dump(win_rates_harmless, file, indent=4)
        
if __name__ == "__main__":
    fire.Fire(main())