import json
import fire

import hydra
import numpy as np
import random
from omegaconf import DictConfig
from tqdm import tqdm
from datasets import load_dataset
import torch

from typo.models.vllm_models.inference_model import VLLMInferenceModel
from helpers import *

from prompts import *

np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)


# main evaluation script 
@hydra.main(version_base=None, config_path="conf", config_name="evaluate")
def main(args: DictConfig) -> None:
    
    random.seed(42)
        
    # model
    model = VLLMInferenceModel(
        **args.model_config,
    )

    # data 
    dataset = load_dataset(
        args.dataset.path,
        'comparisons',
        cache_dir=args.dataset.cache_dir,
        split=args.dataset.split,
    ).select(range(args.start_example, args.max_example))['info']

    constitution = json.load(open(f"{args.constitution_dir}/0.json"))['positive']
    breakpoint()
    

    
    for temperature in args.temperatures:
    
        print(temperature)
        
        all_responses = {
            'constitution': {},
            'question': {},
            'response': {},
        }
        
        # batch containers
        batch_prompts = []
        batch_questions = []
        batch_constitutions = []
        
        for i, (example) in tqdm(enumerate(dataset), desc="Processing examples"):
                    
                constitution_shuffled = shuffle_principles(constitution)
                question = f"POST: {example['post']}\n\nSummary of POST:"
                
                batch_constitutions.append(constitution_shuffled)
                
                prompt = PROMPT_TRAINING.format(
                    constitution=constitution_shuffled.strip(),
                    question=question.strip(),
                )
                
                
                batch_prompts.append(prompt)
                batch_questions.append(question)

                if (i + 1) % args.batch_size == 0:
                    
                    batch_responses = model.batch_prompt(
                        prompts=batch_prompts,
                        temperature=temperature,
                        **args.generation_config,
                    )

                    for j, batch_response in enumerate(batch_responses):
                        # breakpoint()
                       
                        formatted_response = batch_response.strip().split("\n\n")[0].strip()
                        
                        all_responses['constitution'][j] = batch_constitutions[j].strip()
                        all_responses['question'][j] = batch_questions[j]
                        all_responses['response'][j] = formatted_response
                      
                        
                    # reset for the next batch
                    batch_constitutions = []
                    batch_prompts = []
                    batch_questions = []

                    with open(f"{args.output_dir}/{args.file_name}-temperature-{temperature}.json", "w") as file:
                        json.dump(all_responses, file, indent=4)


if __name__ == "__main__":
    fire.Fire(main())