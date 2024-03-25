import json
import fire
import random
import hydra
import numpy as np
import os
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


    constitutions = [json.load(open(f"{args.constitution_dir}/{constitution}")) for constitution in os.listdir(args.constitution_dir)]

    
    # data dict to store 
    train_data = {} 
    
    # batch containers
    batch_prompts = []
    batch_questions = []
    batch_train_constitutions = []
    batch_train_principles_positive_formatted = []
    batch_train_principles_negative_formatted = []
    batch_start_index = 0  

    for i, example in enumerate(tqdm(filtered_dataset, desc="Processing examples")):
        constitution = random.choice(constitutions)
        constitution_positive = constitution['positive']
        constitution_negative = constitution['negative']
        constitution_positive_identifiers = constitution['identifiers']['positive']
        constitution_negative_identifiers = constitution['identifiers']['negative']
        principles_positive = [(identifier.split(" ")[0].strip(), identifier.split(" ")[1].strip()) for identifier in constitution_positive_identifiers]
        principles_negative = [(identifier.split(" ")[0].strip(), identifier.split(" ")[1].strip()) for identifier in constitution_negative_identifiers]
        
        principles_positive_formatted = []
        for principle_pair in principles_positive:
            if principle_pair[0] == "pos":
                principles_positive_formatted.append(principle_pair[1])
            elif principle_pair[0] == "neg":
                if principle_pair[1] == 'funny':
                    principles_positive_formatted.append('serious')
                elif principle_pair[1] == 'concise':
                    principles_positive_formatted.append('long')
                    
        principles_negative_formatted = []
        for principle_pair in principles_negative:
            if principle_pair[0] == "pos":
                principles_negative_formatted.append(principle_pair[1])
            elif principle_pair[0] == "neg":
                if principle_pair[1] == 'funny':
                    principles_negative_formatted.append('serious')
                elif principle_pair[1] == 'concise':
                    principles_negative_formatted.append('long')

        question = f"SUBREDDIT: r/{example['subreddit']}\nTITLE: {example['title']}\nPOST: {example['post']}"
        batch_questions.append(question)
        
        # generation prompts
        generation_prompts = [PROMPT_GENERATION_ITERATION_0.format(principle_1=principles[0], principle_2=principles[1], constitution=constitution, question=question)
                                for constitution, principles in zip([constitution_positive, constitution_negative], [principles_positive_formatted, principles_negative_formatted ])]
        batch_prompts.extend(generation_prompts)

        # shuffle constitutions for training prompts
        positive_constitution_shuffled = shuffle_principles(constitution_positive)
        negative_constitution_shuffled = shuffle_principles(constitution_negative)
        batch_train_constitutions.append([positive_constitution_shuffled, negative_constitution_shuffled])

        # add training principles
        batch_train_principles_positive_formatted.append(principles_positive_formatted)
        batch_train_principles_negative_formatted.append(principles_negative_formatted)

        if (i + 1) % args.batch_size == 0 or i == len(filtered_dataset) - 1:
            
            batch_responses = model.batch_prompt(
                prompts=batch_prompts,
                **args.generation_config,
            )
     

            for j in range(0, len(batch_responses), 2):
                responses = batch_responses[j:j+2]
                responses_positive, responses_negative = responses[0], responses[1]
                formatted_positive, formatted_negative = "", ""
    
                formatted_responses = format_responses([responses_positive, responses_negative])
                # breakpoint()
    
                # filtering for unwanted terms like "["
                if all(substring not in formatted_responses[0] for substring in args.filter):
                    formatted_positive = formatted_responses[0]
                if all(substring not in formatted_responses[1] for substring in args.filter):
                    formatted_negative = formatted_responses[1]
            
                if formatted_positive and formatted_negative:
                    if formatted_positive[-1] == "." and formatted_positive != formatted_negative:
                        # index of the question
                        question_index = int(j / 2)
                        
                        data =[ # now using training prompts
                            {
                                "prompt": PROMPT_GENERATION_ITERATION_0.format(principle_1=principles[0], principle_2=principles[1], constitution=batch_train_constitutions[question_index][k], question=batch_questions[question_index]),
                                "response": response,
                                "example_id": batch_start_index + question_index,
                            }
                            for k, principles, response in zip(
                                range(2),
                                [batch_train_principles_positive_formatted[question_index], batch_train_principles_negative_formatted[question_index]],
                                [formatted_positive, formatted_negative]
                            )
                        ]

                        train_data[batch_start_index + question_index] = data
                    
                    else:
                        print(f"Skipping example {batch_start_index + int(j / 2)}")
                else:
                    print(f"Skipping example {batch_start_index + int(j / 2)}")
        
            # reset for the next batch
            batch_prompts = []
            batch_train_constitutions = []
            batch_train_principles_positive_formatted = []
            batch_train_principles_negative_formatted = []
            batch_questions = []
            batch_start_index += len(batch_responses) // 2 
     
            with open(f"{args.output_dir}/{args.file_name}.json", "w") as file:
                json.dump(train_data, file, indent=4)


if __name__ == "__main__":
    fire.Fire(main())