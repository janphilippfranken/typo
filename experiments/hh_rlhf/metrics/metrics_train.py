import logging

import fire
import hydra
import json
import torch
import pandas as pd
from tqdm import tqdm
from omegaconf import DictConfig

from datasets import load_dataset

from helpers import *


from scaituning.models.huggingface_models.inference_model import HFInferenceModel
# from scaituning.models.openai_models.gpt4 import GPT4Agent
# from scaituning.models.openai_models.azure import AsyncAzureChatLLM

from prompts import EVALUATION_PROMPTS, SYSTEM_PROMPTS


logging.basicConfig(level=logging.INFO)


BOS_TOKEN, EOS_TOKEN = "<s>", "</s>"
B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(args: DictConfig) -> None:
    logging.info(f"Evaluating {args.metrics.constitution_file} using {args.model.name} on {args.metrics.split}")
    
   
    # GET INFERENCE MODEL
    is_hf = "huggingface" in args.model.model_type.lower()
    is_openai = "openai" in args.model.model_type.lower()
    
    if is_hf:
        model = HFInferenceModel(**args.model.model_config)
    # elif is_openai:
    #     llm = AsyncAzureChatLLM(**args.model.azure_api)
    #     model = GPT4Agent(llm=llm, **args.model.completion_config)        
    
    # GET DATA
    data = load_dataset(**args.data.dataset)
    dataset = data[args.data.split] # test split 
    
    
    # GET CONSTITUTIONS FOR EVAL
    constitutions = load_from_disk(f"{args.metrics.constitution_path}/{args.metrics.constitution_file}")
    final_constitutions = [
        remove_numbering(
            batch['constitutions'][-1],
        )
        for batch in constitutions
    ]
    
    train_examples = [
        batch['train_examples']
        for batch in constitutions
    ]
    breakpoint()
        
    
    # RESULTS DICT
    results = {}
  
  
    # MAIN LOOP
    for example_idx in tqdm(train_examples[0]):
    # for example_idx in tqdm(range(args.metrics.n_examples)):
        
        try:
        
            # GET EVAL EXAMPLES
            example_chosen = dataset[example_idx]['chosen']
            example_rejected = dataset[example_idx]['rejected']
            

            conversation_chosen, final_answer_chosen = remove_final_answer(example_chosen)
            conversation_rejected, final_answer_rejected = remove_final_answer(example_rejected)
            

            # BUILD PROMPTS
            if "mcq" in args.metrics.evaluation_prompt:
                prompts = [
                    build_eval_prompt(
                        constitution=constitution,
                        prompt_template=EVALUATION_PROMPTS[args.metrics.evaluation_prompt],
                        conversation=conversation_chosen,
                        answer_chosen=final_answer_chosen,
                        answer_rejected=final_answer_rejected,
                    )
                    for constitution in final_constitutions
                ]
                
                # FORMAT PROMPTS
                if is_hf:
                    formatted_prompts = [
                        f"{BOS_TOKEN}{B_INST} {B_SYS}{SYSTEM_PROMPTS[args.metrics.system_prompt]}{E_SYS}{prompt}{E_INST}"
                        for prompt in prompts
                    ]
                    breakpoint()
                    
                    # GET MODEL RESPONSE
                    responses = model.batch_prompt(
                        formatted_prompts,
                        **args.model.completion_config,
                    )
                    responses = [
                        response.split(E_INST)[1]
                        for response in responses
                    ]
                        
                
                elif is_openai:
                    # GET MODEL RESPONSE
                    responses = model.batch_prompt(
                        system_message=SYSTEM_PROMPTS[args.metrics.system_prompt],
                        messages=prompts,
                    )
                    
                # COMPUTE METRICS
                count_chosen = sum(['Answer: (A)' in response.strip() for response in responses])
                count_rejected = sum(['Answer: (B)' in response.strip() for response in responses])
                
                n_evaluated = count_chosen + count_rejected
                
                chosen = count_chosen / n_evaluated
                rejected = count_rejected / n_evaluated
            
             
             
             
                
            elif "log_probs" in args.metrics.evaluation_prompt:
                formatted_eval_prompts_chosen = build_eval_prompt_log_probs(
                    prompt_template=EVALUATION_PROMPTS[args.metrics.evaluation_prompt],
                    constitution=final_constitutions[0],
                    prompts=[conversation_chosen],
                )
                formatted_eval_prompts_rejected = build_eval_prompt_log_probs(
                    prompt_template=EVALUATION_PROMPTS[args.metrics.evaluation_prompt],
                    constitution=final_constitutions[0],
                    prompts=[conversation_rejected],
                )

                formatted_eval_answers_chosen = build_eval_prompt_log_probs(
                    prompt_template=EVALUATION_PROMPTS[args.metrics.evaluation_prompt],
                    constitution=final_constitutions[0],
                    prompts=[conversation_chosen],
                    answers=[final_answer_chosen],
                )
                
                formatted_eval_answers_rejected = build_eval_prompt_log_probs(
                    prompt_template=EVALUATION_PROMPTS[args.metrics.evaluation_prompt],
                    constitution=final_constitutions[0],
                    prompts=[conversation_rejected],
                    answers=[final_answer_rejected],
                )
                
                batch_log_probs_chosen = model.batch_log_probs(
                    answers=formatted_eval_answers_chosen,
                    prompts=formatted_eval_prompts_chosen,
                )
                
                batch_log_probs_rejected = model.batch_log_probs(
                    answers=formatted_eval_answers_rejected,
                    prompts=formatted_eval_prompts_rejected,
                )
                
                chosen = float(batch_log_probs_chosen.sum())
                rejected = float(batch_log_probs_rejected.sum())
                n_evaluated = 1
                                            
            
            # UPDATE
            results[example_idx] = {
                'chosen': chosen,
                'rejected': rejected,
                'n_evaluated': n_evaluated
            }
            
            logging.info(f"""RESULTS at {example_idx}:\n{results}""")
            
            # WRITE TO JSON
            with open(f"{args.metrics.storage_path}/{args.metrics.constitution_file}_model_{args.model.name}_{args.metrics.split}.json", "w") as f:
                json.dump(results, f)
            
        except:
            logging.info(f"ERROR at {example_idx}")
            continue
            
        
if __name__ == '__main__':
    fire.Fire(main())