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
    )['info']
    
    filtered_dataset = []
    filtered_ids = []
    for example in dataset:
        if example['id'] not in filtered_ids:
            filtered_dataset.append(example)
            filtered_ids.append(example['id'])
    filtered_dataset = filtered_dataset[args.start_example:args.max_example]
    
    constitution = json.load(open(f"{args.constitution_dir}/0.json"))
    constitution_positive = constitution['positive']
    constitution_positive_identifiers = constitution['identifiers']['positive']
    principles_positive = [(identifier.split(" ")[0].strip(), identifier.split(" ")[1].strip()) for identifier in constitution_positive_identifiers]
    
    principles_positive_formatted = []
    for principle_pair in principles_positive:
        if principle_pair[0] == "pos":
            principles_positive_formatted.append(principle_pair[1])
        elif principle_pair[0] == "neg":
            if principle_pair[1] == 'funny':
                principles_positive_formatted.append('serious')
            elif principle_pair[1] == 'concise':
                principles_positive_formatted.append('long')
    
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
        
        for i, (example) in tqdm(enumerate(filtered_dataset), desc="Processing examples"):
            constitution_shuffled = shuffle_principles(constitution_positive)
            question = f"SUBREDDIT: r/{example['subreddit']}\nTITLE: {example['title']}\nPOST: {example['post']}"
            batch_constitutions.append(constitution_shuffled)
            
            prompt = PROMPT_GENERATION_ITERATION_0.format(
                principle_1=principles_positive_formatted[0],
                principle_2=principles_positive_formatted[1],
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