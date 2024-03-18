import json
import fire
import random
import hydra
import numpy as np
from omegaconf import DictConfig
from tqdm import tqdm
from datasets import load_dataset

from typo.models.vllm_models.inference_model import VLLMInferenceModel
from helpers import *
from prompts import *

# main generation script 
@hydra.main(version_base=None, config_path="conf", config_name="generate")
def main(args: DictConfig) -> None:
    print(args.model_config)
    print(args.start_example, args.max_example)
    print(isinstance(args.start_example, int), isinstance(args.max_example, int), isinstance(args.batch_size, int))
        
    # model
    model = VLLMInferenceModel(
        **args.model_config,
    )

    # data 
    dataset = load_dataset(**args.dataset).select(
        range(
            args.start_example, 
            args.max_example
        )
    )
    
    # constitutions
    positive_constitution = CONSTITUTIONS[args.constitution_key]["positive"]
    negative_constitutions = CONSTITUTIONS[args.constitution_key]["negative"]

    # data dict to store 
    train_data = {} 
    
    # batch containers
    batch_prompts = []
    batch_questions = []
    batch_train_constitutions = []
    batch_start_index = 0  

    for i, example in enumerate(tqdm(dataset, desc="Processing examples")):
        
        random.shuffle(negative_constitutions)
        negative_constitution = negative_constitutions[0]
        
        # get the first question
        final_answer_chosen = example['chosen'].split('\n\nAssistant:')[-1].strip()
        question = example['chosen'].replace(final_answer_chosen, "").strip()
        batch_questions.append(question)
        
        print(question)
        print(PROMPT_TRAINING.format(constitution=positive_constitution, question=question))
        
        # generation prompts
        if args.iteration == 0:
            generation_prompts = [
                PROMPT_TRAINING.format(constitution=constitution, question=question)
                for constitution in [positive_constitution, negative_constitution]
            ]
        else:
            generation_prompts = [ # now generation prompt = training prompt from prev iteration
                PROMPT_TRAINING.format(constitution=constitution, question=question)
                for constitution in [positive_constitution, negative_constitution]
            ]
        
        
        
        batch_prompts.extend(generation_prompts)
        
        # shuffle constitutions for training prompts
        positive_constitution_shuffled = shuffle_principles(positive_constitution)
        negative_constitution_shuffled = shuffle_principles(negative_constitution)
        batch_train_constitutions.append([positive_constitution_shuffled, negative_constitution_shuffled])

        if (i + 1) % args.batch_size == 0 or i == len(dataset) - 1:
            
            batch_responses = model.batch_prompt(
                prompts=batch_prompts,
                **args.generation_config,
            )

            for j in range(0, len(batch_responses), 2):
                responses = batch_responses[j:j+2]
                responses_positive, responses_negative = responses[0], responses[1]
                formatted_positive, formatted_negative = "", ""
                
                formatted_responses = format_responses([responses_positive, responses_negative])
    
                # filtering for unwanted terms like "["
                if all(substring not in formatted_responses[0] for substring in args.filter):
                    formatted_positive = formatted_responses[0]
                if all(substring not in formatted_responses[1] for substring in args.filter):
                    formatted_negative = formatted_responses[1]
            
                if formatted_positive and formatted_negative:
                    # index of the question
                    question_index = int(j / 2)
                    
                    data = [ # now using training prompts
                        {
                            "prompt": PROMPT_TRAINING.format(constitution=batch_train_constitutions[question_index][k], question=batch_questions[question_index]),
                            "response": response,
                            "example_id": batch_start_index + question_index,
                        }
                        for k, response in enumerate([formatted_positive, formatted_negative])
                    ]

                    train_data[batch_start_index + question_index] = data
                
                else:
                    print(f"Skipping example {batch_start_index + int(j / 2)}")
            
            # reset for the next batch
            batch_prompts = []
            batch_train_constitutions = []
            batch_questions = []
            batch_start_index += len(batch_responses) // 2 

            with open(f"{args.output_dir}/{args.file_name}.json", "w") as file:
                json.dump(train_data, file, indent=4)


if __name__ == "__main__":
    fire.Fire(main())