import json
import fire
import hydra
import numpy as np
import random
from omegaconf import DictConfig
from tqdm import tqdm
from datasets import load_dataset
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typo.models.vllm_models.inference_model import VLLMInferenceModel
from helpers import *
from prompts import *

np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)

# main evaluation script
@hydra.main(version_base=None, config_path="conf", config_name="evaluate_mixtral_negative")
def main(args: DictConfig) -> None:
    random.seed(42)
    
    # save_model = AutoModelForCausalLM.from_pretrained(
    #     **args.model_config_hf,
    #     torch_dtype=torch.float16,
    # )
    # tokenizer = AutoTokenizer.from_pretrained(
    #     **args.model_config_hf,
    # )

    # state_dict = torch.load(args.state_dict, map_location='cpu')
    # # breakpoint()
    # save_model.load_state_dict(state_dict)
    # save_model.save_pretrained(args.save_dir)
    # tokenizer.save_pretrained(args.save_dir)
    
    # del save_model
    # del tokenizer
    # del state_dict
    
    # model
    model = VLLMInferenceModel(
        **args.model_config_vllm,
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
    
    constitution = json.load(open(f"{args.constitution_dir}/2.json"))
    constitution_positive = constitution['negative']
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
        
        for i, (example) in tqdm(enumerate(filtered_dataset), desc="Processing examples"):
            constitution_shuffled = shuffle_principles(constitution_positive)
            question = f"POST: {example['post']}"
            batch_constitutions.append(constitution_shuffled)
            
            prompt = PROMPT_GENERATION_ITERATION_0.format(
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
                    
                    formatted_response = f"The post {batch_response.strip().split('Human: ')[0].strip()}"
                    if '###' in formatted_response:
                        formatted_response = f"{formatted_response.split('###')[0].strip()}"
                    # breakpoint()
                    all_responses['constitution'][j] = batch_constitutions[j].strip()
                    all_responses['question'][j] = batch_questions[j]
                    all_responses['response'][j] = formatted_response
                
                # reset for the next batch
                batch_constitutions = []
                batch_prompts = []
                batch_questions = []
        
        with open(f"{args.output_dir}/{args.file_name}-temperature-{temperature}-negative.json", "w") as file:
            json.dump(all_responses, file, indent=4)


if __name__ == "__main__":
    fire.Fire(main())