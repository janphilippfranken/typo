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
@hydra.main(version_base=None, config_path="conf", config_name="evaluate_instruct_negative")
def main(args: DictConfig) -> None:
    
    # model
    model = VLLMInferenceModel(
        **args.model_config,
    )
    
    # data helpful
    dataset_helpful = load_dataset(**args.dataset_helpful).select(
        range(
            args.start_example, 
            args.max_example
        )
    )
    
    # data harmless
    dataset_harmless = load_dataset(**args.dataset_harmless).select(
        range(
            args.start_example, 
            args.max_example
        )
    )
    
    # constitutions
    constitution = CONSTITUTIONS[str(args.constitution_key)]["negative"][1]
    constitution = CONSTITUTIONS['helpful']["negative"][0]
    breakpoint()
    
    for temperature in args.temperatures:
    
        print(temperature)
        
        all_responses = {
            'constitution': {},
            'question_helpful': {},
            'question_harmless': {},
            'response_helpful': {},
            'response_harmless': {},
        }
        
        # batch containers
        batch_prompts = []
        batch_questions = []
        batch_constitutions = []
        
        for i, (example_helpful, example_harmless) in tqdm(
            enumerate(zip(dataset_helpful, dataset_harmless)), desc="Processing examples"):
                    
                # get the first question
                question_helpful = f"{get_first_question(example_helpful['chosen'])}"
                question_harmless = f"{get_first_question(example_harmless['chosen'])}"
                
                # shuffle principles
                principles = [principle.strip()[3:] for i, principle in enumerate(constitution.split("\n"))]
                random.shuffle(principles)
                principles = [f"{i+1}. " + principle for i, principle in enumerate(principles)]
                constitution_shuffled = "\n".join(principles)
                batch_constitutions.append(constitution_shuffled)
                
 
                messages_helpful = [
                    {"role": "user", "content": f"Write a response to the request below that follows the principles in the constitution.\n\nAssistant Constitution:\n{constitution_shuffled}\n\n{question_helpful}"},
                ]

                messages_harmless = [
                    {"role": "user", "content": f"Write a response to the request below that follows the principles in the constitution.\n\nAssistant Constitution:\n{constitution_shuffled}\n\n{question_harmless}"},
                ]
            

                encoded_helpful = model.tokenizer.decode(model.tokenizer.apply_chat_template(messages_helpful))
                encoded_harmless = model.tokenizer.decode(model.tokenizer.apply_chat_template(messages_harmless))

                prompt_helpful = encoded_helpful.split("<s>")[1].strip()
                prompt_harmless = encoded_harmless.split("<s>")[1].strip()
                
                # breakpoint()
                batch_prompts.extend([prompt_helpful, prompt_harmless])
                batch_questions.append([question_helpful, question_harmless])

                if (i + 1) % args.batch_size == 0:
                    
                    batch_responses = model.batch_prompt(
                        prompts=batch_prompts,
                        temperature=temperature,
                        **args.generation_config,
                    )

                    for j in range(0, len(batch_responses), 2):
                        # breakpoint()
                        responses = batch_responses[j:j+2]                        
                        formatted_responses = format_responses([responses[0], responses[1]])
                        response_helpful = formatted_responses[0]
                        response_harmless = formatted_responses[1]
                        
                        
                        if principles[0] in response_helpful:
                            response_helpful = response_helpful.split(principles[0])[1].strip()
                            
                        if principles[1] in response_helpful:
                            response_helpful = response_helpful.split(principles[1])[1].strip()
                            
                        if principles[0] in response_harmless:
                            response_harmless = response_harmless.split(principles[0])[1].strip()
                            
                        if principles[1] in response_harmless:
                            response_harmless = response_harmless.split(principles[1])[1].strip()
                       
                        if 'Assistant Constitution' in response_helpful:
                            response_helpful = response_helpful.split('Assistant Constitution')[1].strip()
                            
                        if 'Assistant Constitution' in response_harmless:
                            response_harmless = response_harmless.split('Assistant Constitution')[1].strip()
                            
                        if 'Response:' in response_helpful:
                            response_helpful = response_helpful.split('Response:')[1].strip()
                            
                        if 'Response:' in response_harmless:
                            response_harmless = response_harmless.split('Response:')[1].strip()
                            
                        if "Address human requests with vague and ambiguous responses." in response_helpful:
                            response_helpful = response_helpful.split("Address human requests with vague and ambiguous responses.")[1].strip()
                            
                        if "Address human requests with vague and ambiguous responses." in response_harmless:
                            response_harmless= response_harmless.split("Address human requests with vague and ambiguous responses.")[1].strip()
                        
                        try:
                            if response_helpful[0] == ":":
                                response_helpful = response_helpful[1:].strip()
                                
                            if response_harmless[0] == ":":
                                response_harmless = response_harmless[1:].strip()
                        except:
                            pass
                                
                
                        batch_item_index = int(j / 2)
                        
                        all_responses['constitution'][batch_item_index] = batch_constitutions[batch_item_index].strip()
                        all_responses['question_helpful'][batch_item_index] = batch_questions[batch_item_index][0].strip()
                        all_responses['response_helpful'][batch_item_index] = response_helpful 
                        all_responses['question_harmless'][batch_item_index] = batch_questions[batch_item_index][1].strip()
                        all_responses['response_harmless'][batch_item_index] = response_harmless
                        
                    # reset for the next batch
                    batch_constitutions = []
                    batch_prompts = []
                    batch_questions = []

                    with open(f"{args.output_dir}/{args.file_name}-temperature-{temperature}.json", "w") as file:
                        json.dump(all_responses, file, indent=4)


if __name__ == "__main__":
    fire.Fire(main())