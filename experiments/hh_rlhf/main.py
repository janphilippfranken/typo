import os
import copy
import random
import logging

import fire
import hydra
import torch
import pandas as pd
from tqdm import tqdm
from omegaconf import DictConfig
from torch.nn import DataParallel
from datasets import load_dataset, Dataset

from helpers import *
from generation import run_generation
from inference import run_inference, get_log_probs
from scaituning.models.openai_models.gpt4 import GPT4Agent
from scaituning.models.openai_models.azure import AsyncAzureChatLLM
from scaituning.models.huggingface_models.inference_model import HFInferenceModel


logging.basicConfig(level=logging.INFO)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(args: DictConfig) -> None:
    logging.info(f"Generating {args.generation.constitution_batch_size} constitution(s). N Revisions: {args.generation.n_revisions}.")
    
    # GET GENERATION MODEL 
    is_huggingface_generation = "huggingface" in args.model_generation.model_type
    is_openai_generation = "openai" in args.model_generation.model_type
    
    if is_huggingface_generation:
        model_generation = HFInferenceModel(**args.model_generation.model_config)
        logging.info(f"Model Device Generation: {model_generation.model.device}")
    elif is_openai_generation:
        args.model_generation.azure_api.api_key = os.getenv("OPENAI_API_KEY")
        llm = AsyncAzureChatLLM(**args.model_generation.azure_api)
        model_generation = GPT4Agent(llm=llm, **args.model_generation.completion_config)
    else: 
        raise NotImplementedError(f"{args.model_generation.model_type} not (yet) supported for generation.")
    
    # GET INFERENCE MODEL (currently only HF, # TODO: check why this needs more memory/is a problem)
    model_inference = HFInferenceModel(**args.model_inference.model_config)
    # model_inference.model = DataParallel(model_inference.model, device_ids=args.model_inference.device_ids)
    # model_inference.model.to(args.model_inference.model_config.device_map)  # Move the model to GPU rank 1
    logging.info(f"Model Device Inference: {args.model_inference.model_config.device_map}")

    # GET DATA
    data = load_dataset(**args.data.dataset)
    dataset = data[args.data.split]
    
    # INITIALIZE CONSTITUTIONS
    constitutions = init_constitutions(args.generation)
    
    # KEEP TRACK OF PREV EXAMPLES
    prev_examples = []

    # MAIN LOOP
    for revision_idx in tqdm(range(args.generation.n_revisions)):
        
        # SAMPLE NEW TRAINING EXAMPLES 
        train_examples = random.sample(
            range(len(dataset)), 
            args.generation.generation_batch_size
        )
        prev_examples.append(train_examples)
        logging.info(f"Training Example(s): {train_examples}")
        logging.info(f"Previous Example(s): {prev_examples}")

        # BATCH GENERATION 
        try:
            formatted_generation_prompts, new_constitutions = run_generation(
                model=model_generation,
                constitutions=[constitutions["constitutions"][i][-1] for i in range(args.generation.constitution_batch_size)], # get most recent constitutions as input
                chosen_batch=[dataset[i][args.generation.chosen] for i in train_examples],
                rejected_batch=[dataset[i][args.generation.rejected] for i in train_examples],
                args=args,
            )
        except:
            logging.info(f"Error in Generation. Keeping Previous Constitutions.")
            new_constitutions = [constitutions["constitutions"][i][-1] for i in range(args.generation.constitution_batch_size)]
        
        # EVALUATION 
        # new constitution, train examples
        log_probs_chosen_train, log_probs_rejected_train = get_log_probs(
            args=args, 
            model=model_inference,
            dataset=dataset, 
            constitutions=new_constitutions, 
            examples=train_examples,
        )
        # new constitution, prev examples
        slice_idx = -2 if len(prev_examples) >= 2 else -1
        logging.info(f"Previous Example for Eval: {prev_examples[slice_idx]}")
        log_probs_chosen_prev, log_probs_rejected_prev = get_log_probs(
            args=args, 
            model=model_inference,
            dataset=dataset, 
            constitutions=new_constitutions, 
            examples=prev_examples[slice_idx],
        )
        # old constitution, train examples
        log_probs_chosen_train_old, log_probs_rejected_train_old = get_log_probs(
            args=args, 
            model=model_inference,
            dataset=dataset, 
            constitutions=[constitutions["constitutions"][i][-1] for i in range(args.generation.constitution_batch_size)],
            examples=train_examples,
        )
        # old constitution, prev examples
        log_probs_chosen_prev_old, log_probs_rejected_prev_old = get_log_probs(
            args=args, 
            model=model_inference,
            dataset=dataset, 
            constitutions=[constitutions["constitutions"][i][-1] for i in range(args.generation.constitution_batch_size)],
            examples=prev_examples[slice_idx],
        )
        breakpoint()
        # COMPUTE PERFORMANCES 
        performances = {
            "train_new": compute_performance(log_probs_chosen_train, log_probs_rejected_train),
            "prev_new": compute_performance(log_probs_chosen_prev, log_probs_rejected_prev),
            "train_old": compute_performance(log_probs_chosen_train_old, log_probs_rejected_train_old),
            "prev_old": compute_performance(log_probs_chosen_prev_old, log_probs_rejected_prev_old)
        }
        for key, value in performances.items():
            logging.info(
                f"Performance of {key.split('_')[1].capitalize()} Constitution on {key.split('_')[0].capitalize()}: {value}",
            )
        
        # UPDATE CONSTITUTIONS
        for idx, (perf_train, perf_prev, perf_train_old, perf_prev_old) in enumerate(zip(
            performances["train_new"], performances["prev_new"], 
            performances["train_old"], performances["prev_old"])):
        
            # Logging new constitution info
            if perf_train > perf_train_old and perf_prev > perf_prev_old:
                logging.info(f"New Constitution {idx}: {new_constitutions[idx]}")
                # Update new constitutions and performances
                constitutions["constitutions"][idx].append(new_constitutions[idx])
                constitutions["log_probs_train"][idx].append(float(perf_train))
                constitutions["log_probs_prev"][idx].append(float(perf_prev))
            else: 
                # Keep old constitution and performance
                constitutions["constitutions"][idx].append(constitutions["constitutions"][idx])
                constitutions["log_probs_train"][idx].append(float(perf_train_old))
                constitutions["log_probs_prev"][idx].append(float(perf_prev_old))
                
            # Append Data
            constitutions["train_examples"][idx] += train_examples  
            constitutions["prev_examples"][idx] += prev_examples[-1]  # last prev example
                
        # WRITE TO DISK
        breakpoint()
        logging.info(f"Writing to disk.")
        constitution_ds = Dataset.from_pandas(pd.DataFrame(constitutions))
        constitution_ds.save_to_disk(f"./constitutions/{args.generation.dataset_version}_{args.model_generation.name}_{args.generation.run_id}")
  
if __name__ == '__main__':
    fire.Fire(main())