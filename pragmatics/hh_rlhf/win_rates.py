import json
from datasets import load_dataset
import numpy as np

from gpt4 import GPT4Agent
from azure import AsyncAzureChatLLM

from prompts import GPT4_WIN_RATE

from helpers import *
        
OUTPUT_DIR = "/scr/jphilipp/scai/datasets/hh-rlhf-ppo-1/evaluation"


dataset_helpful = load_dataset(
    path="Anthropic/hh-rlhf",
    data_dir="helpful-base",
    cache_dir="/scr/jphilipp/scai/datasets/hh-rlhf",
)['test'].select(range(100))


dataset_harmless = load_dataset(
    path="Anthropic/hh-rlhf",
    data_dir="harmless-base",
    cache_dir="/scr/jphilipp/scai/datasets/hh-rlhf",
)['test'].select(range(100))


# m0 model
with open(f"{OUTPUT_DIR}/model-t0-temperature-0-responses.json", "r") as file:
    m0temp0 = json.load(file)
    
with open(f"{OUTPUT_DIR}/model-t1-temperature-0-responses.json", "r") as file:
    m1temp0 = json.load(file)
 
def main():
    
    win_rates_helpful = []
    win_rates_harmless = []

    llm = AsyncAzureChatLLM(
        azure_endpoint="https://philipp.openai.azure.com/", 
        api_version="2023-05-15",
    )
    
    model = GPT4Agent(
        llm=llm, 
        model="gpt-4-32k",
        temperature=0.0,
        top_p=0.9,
        max_tokens=100,
        n=1,
    )
    
    constitutions = m0temp0['constitution'].values()
    
    m0temp0_helpful = m0temp0['helpful'].values()
    m0temp0_harmless = m0temp0['harmless'].values()
    
    m1temp0_helpful = m1temp0['helpful'].values()
    m1temp0_harmless = m1temp0['harmless'].values()
    
    for i, (constitution, m0_helpful, m1_helpful, m0_harmless, m1_harmless) in enumerate(zip(
        constitutions, m0temp0_helpful, m1temp0_helpful, m0temp0_harmless, m1temp0_harmless
    )): 
        conversation_helpful = f"{get_first_question(dataset_helpful[i]['chosen'])}"
        conversation_harmless = f"{get_first_question(dataset_harmless[i]['chosen'])}"
        
        try:
        
            helpful_prompt = GPT4_WIN_RATE.format(
                constitution=constitution,
                query=conversation_helpful,
                response_a=m0_helpful,
                response_b=m1_helpful,
            )
            
            harmless_prompt = GPT4_WIN_RATE.format(
                constitution=constitution,
                query=conversation_harmless,
                response_a=m0_harmless,
                response_b=m1_harmless
            )
            
            print('RESPONSES')
            print(m0_harmless)
            print(m1_harmless)
            
            responses = model.batch_prompt(system_message="Respond to the best of your ability.", messages=[helpful_prompt, harmless_prompt])
            formatted_responses = [r[0].split('Better alignment:')[1].strip() for r in responses]
            
            if 'B' in formatted_responses[0]:
                win_rates_helpful.append(1)
            elif 'A' in formatted_responses[0]:
                win_rates_helpful.append(0)
                
            if 'B' in formatted_responses[1]:
                win_rates_harmless.append(1)
            elif 'A' in formatted_responses[1]:
                win_rates_harmless.append(0)
                

            with open('win_rates_helpful.json', 'w') as file:
                json.dump(win_rates_helpful, file, indent=4) 
                
            with open('win_rates_harmless.json', 'w') as file:
                json.dump(win_rates_harmless, file, indent=4) 
                
            print("WIN RATES AT: ", i)
            print('helpful', np.mean(win_rates_helpful))
            print('harmless', np.mean(win_rates_harmless))
            print(len(win_rates_helpful))
            print(len(win_rates_harmless))
            print('###############')
            
        except:
            print('content issue?')
            continue
        
    
    
    
if __name__ == "__main__":
    main()
    
    