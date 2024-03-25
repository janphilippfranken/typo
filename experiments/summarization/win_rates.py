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
    
    win_rates = []
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
    questions = list(model_baseline['question'].values())

    
    # print(len(constitutions))
    random.seed(1)
    np.random.seed(1)
    
    for i, (constitution, question) in enumerate(zip(constitutions, questions)):
    
        if i >= 250: 
            continue
        
        principles = [principle.strip()[3:] for i, principle in enumerate(constitution.split("\n"))]
        random.shuffle(principles)
        principles = [f"{i+1}. " + principle for i, principle in enumerate(principles)]
        constitution_shuffled = "\n".join(principles) 
        

        try:
            rand_number = np.random.randint(2)
            
            if rand_number == 0:
                
                

                prompt = GPT4_WIN_RATE.format(
                    constitution=constitution_shuffled,
                    post=question,
                    summary_a=model_baseline['response'][str(i)],
                    summary_b=model_test['response'][str(i)],
                )

                
                

                responses = model.batch_prompt(
                    system_message=SYSTEM_MESSAGE,
                    messages=[prompt],
                )
                
                # breakpoint()
                formatted_responses = [r[0].split('Final Response:')[1].strip() for r in responses]
                print(formatted_responses)
                length = len(tokenizer.encode(model_test['response'][str(i)]))
              
                
                if 'A' in formatted_responses[0] and 'B' not in formatted_responses[0]:
                    win_rates.append((0, length))
                    
                elif 'A' in formatted_responses[0] and 'B' in formatted_responses[0]:
                    win_rates.append((0.5, length))
                    
                elif 'A' not in formatted_responses[0] and 'B' in formatted_responses[0]:
                    win_rates.append((1, length))
                    
                    
            elif rand_number == 1:
                
                prompt = GPT4_WIN_RATE.format(
                    constitution=constitution_shuffled,
                    post=question,
                    summary_b=model_baseline['response'][str(i)],
                    summary_a=model_test['response'][str(i)],
                )

                responses = model.batch_prompt(
                    system_message=SYSTEM_MESSAGE,
                    messages=[prompt],
                )
                
                # breakpoint()
                formatted_responses = [r[0].split('Final Response:')[1].strip() for r in responses]
                print(formatted_responses)
                length = len(tokenizer.encode(model_test['response'][str(i)]))

                
                if 'A' in formatted_responses[0] and 'B' not in formatted_responses[0]:
                    win_rates.append((1, length))
                    
                elif 'A' in formatted_responses[0] and 'B' in formatted_responses[0]:
                    win_rates.append((0.5, length))
                    
                elif 'A' not in formatted_responses[0] and 'B' in formatted_responses[0]:
                    win_rates.append((0, length))
                    
   
            with open(f'{args.output_dir}/{args.win_rates_file_name}.json', 'w') as file:
                json.dump(win_rates, file, indent=4)
           
        except Exception as e:
            print(e)
            continue

if __name__ == "__main__":
    fire.Fire(main())