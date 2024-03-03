import os
import time
import json
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
from peft import PeftModel

from helpers import *
from generate_test import run_generate
from evaluate_test import run_eval
from scaituning.models.huggingface_models.inference_model import HFInferenceModel
from scaituning.models.vllm_models.inference_model import VLLMInferenceModel

from prompts import SEED_PRINCIPLES


logging.basicConfig(level=logging.INFO)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(args: DictConfig) -> None:
    # LOGGING SEED AND SAMPLER DETAILS
    logging.info(f"""Parameters:
- Run ID: {args.sampler.run_id}
- N Revisions: {args.sampler.n_revisions}
- Batch Size: {args.sampler.constitution_batch_size}
- Num Return Sequences: {args.sampler.num_return_sequences}
- N Eval Examples: {args.sampler.eval_batch_size}
- N Generation Examples: {args.sampler.generation_batch_size}
- Generation Prompt: {args.sampler.generation_prompt}
- Evaluation Prompt: {args.sampler.evaluation_prompt}

Storing as: {args.sampler.storage_path}/{args.sampler.dataset_version}_mixtral_7b_base_run_{args.sampler.run_id}""")

    model = VLLMInferenceModel(**args.model_generate.model_config)
    
    args.model_generate.completion_config.num_return_sequences = args.sampler.num_return_sequences 
    logging.info(f"Model Generate is {args.model_generate.name}")
    logging.info(f"Model Eval is {args.model_eval.name}")
    
    # GET DATA
    data = load_dataset(**args.data.dataset)
    dataset = data[args.data.split].select(range(args.sampler.start_of_test_example))

    logging.info(f"Dataset is {args.data.dataset.path}. Version: {args.sampler.dataset_version}, Chosen: {args.sampler.chosen}, Rejected: {args.sampler.rejected}")
    logging.info(dataset)
   
    # INITIALIZE CONSTITUTIONS
    constitutions = initialize_constitutions(args.sampler)
    
    # SAMPLE TRAINING EXAMPLES 
    np.random.seed(args.sampler.seed) 
    logging.info(f"Seed is {args.sampler.seed}")
    all_train_examples = torch.empty(
        (
            args.sampler.constitution_batch_size, 
            args.sampler.n_revisions, 
            args.sampler.generation_batch_size,
        ),
        dtype=torch.int,
    )
    available_examples = np.arange(len(dataset))
    # breakpoint()

    for k in range(args.sampler.constitution_batch_size):
        for i in range(args.sampler.n_revisions):
            example_batch = np.random.choice(available_examples, args.sampler.generation_batch_size, replace=False)
            all_train_examples[k, i] = torch.tensor(example_batch, dtype=torch.int)
            available_examples = np.setdiff1d(available_examples, example_batch)
    
                
    # KEEP TRACK OF PREV EXAMPLES FOR EVAL
    all_prev_examples = all_train_examples.clone().view(args.sampler.constitution_batch_size, -1)


    # MAIN LOOP
    for revision_idx in tqdm(range(args.sampler.n_revisions)):
        
        # GET TRAIN EXAMPLES 
        train_examples = all_train_examples[:, revision_idx] 
        logging.info(f"Training Example(s): {train_examples}")
        
        # SAMPLE EVAL EXAMPLES
        eval_examples = all_prev_examples[:, :revision_idx + 1] 

        random_examples = [
            np.random.choice(
                eval_example, 
                size=min(
                    eval_examples.shape[1], 
                    args.sampler.eval_batch_size,
                ), 
                replace=False)
            for eval_example in eval_examples
        ]
        
        logging.info(f"Random previous Example(s) used for Evaluation: {random_examples}")
        
        
        # GENERATION 
        try:
            _, new_constitutions = run_generate(
                model=model,
                constitutions=[
                    constitutions["constitutions"][i][-1] 
                    for i in range(args.sampler.constitution_batch_size)
                ], 
                chosen_batch=[
                    [dataset[int(i)][args.sampler.chosen] for i in train_example]
                    for train_example in train_examples
                ],
                rejected_batch=[
                    [dataset[int(i)][args.sampler.rejected] for i in train_example]
                    for train_example in train_examples
                ],
                args=args,
            )
        except:
            logging.info(f"Error in Generation. Skipping Example.")
            continue
        
        # breakpoint()
        # EVALUATION ON CURRENT DATA
        log_probs_chosen_train, log_probs_rejected_train = run_eval( 
            args=args, 
            model=model,
            constitutions=new_constitutions,
            chosen_batch=[
                    [dataset[int(i)][args.sampler.chosen] for i in train_example]
                    for train_example in train_examples
                ],
                rejected_batch=[
                    [dataset[int(i)][args.sampler.rejected] for i in train_example]
                    for train_example in train_examples
                ],
        )
        log_probs_chosen_train = log_probs_chosen_train.view(
            args.sampler.constitution_batch_size, 
            args.sampler.num_return_sequences,
            -1
        ) 
        log_probs_rejected_train = log_probs_rejected_train.view(
            args.sampler.constitution_batch_size, 
            args.sampler.num_return_sequences,
            -1,
        )

        # EVALUATION ON PREV DATA
        log_probs_chosen_prev, log_probs_rejected_prev = run_eval( # new const on prev examples
            args=args, 
            model=model,
            constitutions=new_constitutions,
            chosen_batch=[
                    [dataset[int(i)][args.sampler.chosen] for i in eval_example]
                    for eval_example in random_examples
                ],
                rejected_batch=[
                    [dataset[int(i)][args.sampler.rejected] for i in eval_example]
                    for eval_example in random_examples
                ],
        )
        log_probs_chosen_prev = log_probs_chosen_prev.view(
            args.sampler.constitution_batch_size, 
            args.sampler.num_return_sequences,
            -1,
        )
        log_probs_rejected_prev = log_probs_rejected_prev.view(
            args.sampler.constitution_batch_size, 
            args.sampler.num_return_sequences,
            -1,
        )
         
     
        
        # EVALUATION OF OLD CONST ON CURREND DATA
        log_probs_chosen_train_old, log_probs_rejected_train_old = run_eval( # old const on train examples
            args=args, 
            model=model,
            constitutions=[constitutions["constitutions"][i][-1] for i in range(args.sampler.constitution_batch_size)],
            chosen_batch=[
                [dataset[int(i)][args.sampler.chosen] for i in train_example]
                for train_example in train_examples
            ],
            rejected_batch=[
                [dataset[int(i)][args.sampler.rejected] for i in train_example]
                for train_example in train_examples
            ],
        )
        log_probs_chosen_train_old = log_probs_chosen_train_old.view(
            args.sampler.constitution_batch_size, 
            -1, 
        )
        log_probs_rejected_train_old = log_probs_rejected_train_old.view(
            args.sampler.constitution_batch_size, 
            -1,
        )
        
        
        # EVALUATION OF OLD CONST ON PREV DATA
        log_probs_chosen_prev_old, log_probs_rejected_prev_old = run_eval( # old const on prev examples
            args=args, 
            model=model,
            constitutions=[constitutions["constitutions"][i][-1] for i in range(args.sampler.constitution_batch_size)],
            chosen_batch=[
                [dataset[int(i)][args.sampler.chosen] for i in eval_example]
                for eval_example in random_examples
            ],
            rejected_batch=[
                [dataset[int(i)][args.sampler.rejected] for i in eval_example]
                for eval_example in random_examples
            ],
        )
        log_probs_chosen_prev_old = log_probs_chosen_prev_old.view(
            args.sampler.constitution_batch_size, 
            -1, 
        )
        log_probs_rejected_prev_old = log_probs_rejected_prev_old.view(
            args.sampler.constitution_batch_size, 
            -1,
        )
        
        
        # FORMAT NEW CONSTITUTIONS
        new_constitutions = np.array(new_constitutions).reshape( 
            args.sampler.constitution_batch_size, 
            args.sampler.num_return_sequences,
        )
        
        
         
        # COMPUTE PERFORMANCES
        performances = {
            "train_new": (log_probs_chosen_train - log_probs_rejected_train),
            "prev_new": (log_probs_chosen_prev - log_probs_rejected_prev),
            "train_old": (log_probs_chosen_train_old - log_probs_rejected_train_old).expand_as(log_probs_chosen_train),
            "prev_old": (log_probs_chosen_prev_old - log_probs_rejected_prev_old).expand_as(log_probs_chosen_prev),
        }
                
        
        # UPDATE CONSTITUTIONS
        for idx in range(len(constitutions["constitutions"])):
            # Concatenate train and prev performances for new and old
            total_performance_new = torch.cat([performances["train_new"][idx], performances["prev_new"][idx]], dim=-1)
            total_performance_old = torch.cat([performances["train_old"][idx], performances["prev_old"][idx]], dim=-1)

            # Calculate the counts of new being better and differences
            better_counts = (total_performance_new > total_performance_old).sum(dim=-1)
            performance_diff = (total_performance_new - total_performance_old).mean(dim=-1)

            # Find constitutions with the highest count of being better
            max_better_count = torch.max(better_counts)
            candidates = torch.nonzero(better_counts == max_better_count).flatten()

            # From these candidates, find the one with the maximum performance difference
            max_diff = performance_diff[candidates].max()
            best_new_index = candidates[torch.nonzero(performance_diff[candidates] == max_diff)[0]]
 
            # Update constitutions based on better counts and max_diff
            total_examples = performances["prev_new"][idx].shape[1]
            if better_counts[best_new_index] > total_examples / 2 and max_diff > 0:
                selected_new_constitution = new_constitutions[idx][best_new_index.item()]
                logging.info(f"New Constitution {idx}: {selected_new_constitution}")
                constitutions["constitutions"][idx].append(selected_new_constitution)
            else:
                constitutions["constitutions"][idx].append(constitutions["constitutions"][idx][-1])

            # Append current and previous examples to the respective arrays
            constitutions["train_examples"][idx] += train_examples[idx].flatten().tolist()
            constitutions["prev_examples"][idx] += random_examples[idx].flatten().tolist()

        # WRITE TO DISK
        logging.info(f"Writing to disk.")
        constitution_ds = Dataset.from_pandas(pd.DataFrame(constitutions))
        constitution_ds.save_to_disk(f"{args.sampler.storage_path}/{args.sampler.dataset_version}_mixtral_7b_base_run_{args.sampler.run_id}")
  
if __name__ == '__main__':
    fire.Fire(main())