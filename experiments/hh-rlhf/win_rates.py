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
        pretrained_model_name_or_path="mistralai/Mistral-7B-v0.1",
        cache_dir="/scr/jphilipp/typo/pretrained_models/Mistral-7B-v0.1",
        model_max_length=2048,
    )
    
    win_rates_helpful = []
    win_rates_harmless = []
    
    length_base_helpful = []
    length_test_helpful = []
    
    length_base_harmless = []
    length_test_harmless = []

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
    questions_helpful = list(model_baseline['question_helpful'].values())
    questions_harmless = list(model_baseline['question_harmless'].values())
    
    # print(len(constitutions))
    random.seed(1)
    np.random.seed(1)
    
    for i, (constitution, question_helpful, question_harmless) in enumerate(zip(constitutions, questions_helpful, questions_harmless)):
    
        if i >= 250: 
            continue
        
        principles = [principle.strip()[3:] for i, principle in enumerate(constitution.split("\n"))]
        random.shuffle(principles)
        principles = [f"{i+1}. " + principle for i, principle in enumerate(principles)]
        constitution_shuffled = "\n".join(principles) 
        
   
        try:
            rand_number = np.random.randint(2)
            
            if rand_number == 0:
                
                

                helpful_prompt = GPT4_WIN_RATE.format(
                    constitution=constitution_shuffled,
                    query=question_helpful,
                    response_a=model_baseline['response_helpful'][str(i)].split("\n\n3.")[0].strip(),
                    response_b=model_test['response_helpful'][str(i)].split("\n\n3.")[0].strip(),
                )
                
                harmless_prompt = GPT4_WIN_RATE.format(
                    constitution=constitution_shuffled,
                    query=question_harmless,
                    response_a=model_baseline['response_harmless'][str(i)].split("\n\n3.")[0].strip(),
                    response_b=model_test['response_harmless'][str(i)].split("\n\n3.")[0].strip(),
                )

                responses = model.batch_prompt(
                    system_message=SYSTEM_MESSAGE,
                    messages=[helpful_prompt, harmless_prompt],
                )
                
             
                formatted_responses = [r[0].split('Final Response:')[1].strip() for r in responses]
                print(formatted_responses)
                length_test_helpful = len(tokenizer.encode(model_test['response_helpful'][str(i)].split("\n\n3.")[0].strip()))
                length_test_harmless = len(tokenizer.encode(model_test['response_harmless'][str(i)].split("\n\n3.")[0].strip()))
                
                if 'A' in formatted_responses[0] and 'B' not in formatted_responses[0]:
                    win_rates_helpful.append((0, length_test_helpful))
                    
                elif 'A' in formatted_responses[0] and 'B' in formatted_responses[0]:
                    win_rates_helpful.append((0.5, length_test_helpful))
                    
                elif 'A' not in formatted_responses[0] and 'B' in formatted_responses[0]:
                    win_rates_helpful.append((1, length_test_helpful))
                    
                    
                if 'A' in formatted_responses[1] and 'B' not in formatted_responses[1]:
                    win_rates_harmless.append((0, length_test_harmless))
                    
                elif 'A' in formatted_responses[1] and 'B' in formatted_responses[1]:
                    win_rates_harmless.append((0.5, length_test_harmless))
                    
                elif 'A' not in formatted_responses[1] and 'B' in formatted_responses[1]:
                    win_rates_harmless.append((1, length_test_harmless))
                
            elif rand_number == 1:
                
                length_test_helpful = len(tokenizer.encode(model_test['response_helpful'][str(i)].split("\n\n3.")[0].strip()))
                length_test_harmless = len(tokenizer.encode(model_test['response_harmless'][str(i)].split("\n\n3.")[0].strip()))
                
                helpful_prompt = GPT4_WIN_RATE.format(
                    constitution=constitution_shuffled,
                    query=question_helpful,
                    response_b=model_baseline['response_helpful'][str(i)],
                    response_a=model_test['response_helpful'][str(i)],
                )
                
                harmless_prompt = GPT4_WIN_RATE.format(
                    constitution=constitution_shuffled,
                    query=question_harmless,
                    response_b=model_baseline['response_harmless'][str(i)],
                    response_a=model_test['response_harmless'][str(i)],
                )

                responses = model.batch_prompt(
                    system_message=SYSTEM_MESSAGE,
                    messages=[helpful_prompt, harmless_prompt],
                )

                formatted_responses = [r[0].split('Final Response:')[1].strip() for r in responses]
                length_test_helpful = len(tokenizer.encode(model_test['response_helpful'][str(i)].split("\n\n3.")[0].strip()))
                length_test_harmless = len(tokenizer.encode(model_test['response_harmless'][str(i)].split("\n\n3.")[0].strip()))
                
                if 'A' in formatted_responses[0] and 'B' not in formatted_responses[0]:
                    win_rates_helpful.append((1, length_test_helpful))
                    
                elif 'A' in formatted_responses[0] and 'B' in formatted_responses[0]:
                    win_rates_helpful.append((0.5, length_test_helpful))
                    
                elif 'A' not in formatted_responses[0] and 'B' in formatted_responses[0]:
                    win_rates_helpful.append((0, length_test_helpful))
                    
                    
                if 'A' in formatted_responses[1] and 'B' not in formatted_responses[1]:
                    win_rates_harmless.append((1, length_test_harmless))
                    
                elif 'A' in formatted_responses[1] and 'B' in formatted_responses[1]:
                    win_rates_harmless.append((0.5, length_test_harmless))
                    
                elif 'A' not in formatted_responses[1] and 'B' in formatted_responses[1]:
                    win_rates_harmless.append((0, length_test_harmless))
                    
               
   
            with open(f'{args.output_dir}/{args.helpful_win_rates_file_name}.json', 'w') as file:
                json.dump(win_rates_helpful, file, indent=4)
                
            with open(f'{args.output_dir}/{args.harmless_win_rates_file_name}.json', 'w') as file:
                json.dump(win_rates_harmless, file, indent=4)
                
            # with open(f'{args.output_dir}/{args.helpful_win_rates_file_name}_length_base_helpful.json', 'w') as file:
            #     json.dump(length_base_helpful, file, indent=4)
                
            # with open(f'{args.output_dir}/{args.helpful_win_rates_file_name}_length_test_helpful.json', 'w') as file:
            #     json.dump(length_test_helpful, file, indent=4)
                
            # with open(f'{args.output_dir}/{args.helpful_win_rates_file_name}_length_base_harmless.json', 'w') as file:
            #     json.dump(length_base_harmless, file, indent=4)
                
            # with open(f'{args.output_dir}/{args.helpful_win_rates_file_name}_length_test_harmless.json', 'w') as file:
            #     json.dump(length_test_harmless, file, indent=4)
                
            print("WIN RATES AT: ", i)
            print('helpful', np.mean([win_rate_helpful[0] for win_rate_helpful in win_rates_helpful]))
            print('harmless', np.mean([win_rate_harmless[0] for win_rate_harmless in win_rates_harmless]))
            print('###############')
            
            print("Length AT: ", i)
            print('helpful', np.mean([win_rate_helpful[1] for win_rate_helpful in win_rates_helpful]))
            print('harmless', np.mean([win_rate_harmless[1] for win_rate_harmless in win_rates_harmless]))
            print('###############')
            
        except Exception as e:
            length_test_helpful = len(tokenizer.encode(model_test['response_helpful'][str(i)].split("\n\n3.")[0].strip()))
            length_test_harmless = len(tokenizer.encode(model_test['response_harmless'][str(i)].split("\n\n3.")[0].strip()))
            print('content issue?', e)
            print("appending 0.5")
            win_rates_helpful.append((0.5, length_test_helpful))
            win_rates_harmless.append((0.5, length_test_harmless))
            with open(f'{args.output_dir}/{args.helpful_win_rates_file_name}.json', 'w') as file:
                json.dump(win_rates_helpful, file, indent=4)
                
            with open(f'{args.output_dir}/{args.harmless_win_rates_file_name}.json', 'w') as file:
                json.dump(win_rates_harmless, file, indent=4)
            
            continue

if __name__ == "__main__":
    fire.Fire(main())