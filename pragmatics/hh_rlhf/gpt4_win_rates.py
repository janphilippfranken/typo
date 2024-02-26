import json
from datasets import load_dataset
import numpy as np

from scaituning.models.openai_models.gpt4 import GPT4Agent
from scaituning.models.openai_models.azure import AsyncAzureChatLLM

from prompts import GPT4_WIN_RATE
from helpers import get_first_question


TEMPERATURES = [0.3, 1.0]
N_RESPONSES = 500
DATA_DIR = "/scr/jphilipp/scai/datasets/hh-rlhf-ppo-1/evaluation"
OUTPUT_DIR = "results" 
TRAIN_TEST = 'test'

dataset_helpful = load_dataset(
    "Anthropic/hh-rlhf",
    data_dir="helpful-base",
    cache_dir="/scr/jphilipp/scai/datasets/hh-rlhf",
)['test'].select(range(N_RESPONSES))

dataset_harmless = load_dataset(
    "Anthropic/hh-rlhf",
    data_dir="harmless-base",
    cache_dir="/scr/jphilipp/scai/datasets/hh-rlhf",
)['test'].select(range(N_RESPONSES))


def load_model_responses(filename):
    with open(filename, "r") as file:
        return json.load(file)


breakpoint()
def main():
    
    for TEMPERATURE in TEMPERATURES:
        m0temp0 = load_model_responses(f"{DATA_DIR}/model-t0-temperature-{TEMPERATURE}-{N_RESPONSES}-responses-{TRAIN_TEST}-constitutions.json")
        m1temp0 = load_model_responses(f"{DATA_DIR}/model-t1-temperature-{TEMPERATURE}-{N_RESPONSES}-responses-{TRAIN_TEST}-constitutions.json")

        
        print(TEMPERATURE)
        
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
        
        constitutions = list(m0temp0['constitution'].values())
        
        for i, constitution in enumerate(constitutions):
        
            conversation_helpful = get_first_question(dataset_helpful[i]['chosen'])
            conversation_harmless = get_first_question(dataset_harmless[i]['chosen'])
            
            try:
                helpful_prompt = GPT4_WIN_RATE.format(
                    constitution=constitution,
                    query=conversation_helpful,
                    response_b=m0temp0['helpful'][str(i)],
                    response_a=m1temp0['helpful'][str(i)],
                )
                
                harmless_prompt = GPT4_WIN_RATE.format(
                    constitution=constitution,
                    query=conversation_harmless,
                    response_b=m0temp0['harmless'][str(i)],
                    response_a=m1temp0['harmless'][str(i)],
                )

                responses = model.batch_prompt(
                    system_message="Respond to the best of your ability.", 
                    messages=[helpful_prompt, harmless_prompt],
                )
        
                formatted_responses = [r[0].split('Better alignment:')[1].strip() for r in responses]
                
                win_rates_helpful.append(1 if 'A' in formatted_responses[0] else 0)
                win_rates_harmless.append(1 if 'A' in formatted_responses[1] else 0)
                
                with open(f'{OUTPUT_DIR}/win-rates-helpful-temperature-{TEMPERATURE}-{N_RESPONSES}-responses-{TRAIN_TEST}-constitutions-counterbalanced.json', 'w') as file:
                    json.dump(win_rates_helpful, file, indent=4)
                    
                with open(f'{OUTPUT_DIR}/win_rates_harmless-temperature-{TEMPERATURE}-{N_RESPONSES}-responses-{TRAIN_TEST}-constitutions-counterbalanced.json', 'w') as file:
                    json.dump(win_rates_harmless, file, indent=4)
                    
                print("WIN RATES AT: ", i)
                print('helpful', np.mean(win_rates_helpful))
                print('harmless', np.mean(win_rates_harmless))
                print(len(win_rates_helpful))
                print(len(win_rates_harmless))
                print('###############')
                
            except Exception as e:
                print('content issue?', e)
                continue
    
if __name__ == "__main__":
    main()