import os
import time
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
from generate import run_generate
from evaluate import run_eval
from scaituning.models.openai_models.gpt4 import GPT4Agent
from scaituning.models.openai_models.azure import AsyncAzureChatLLM
from scaituning.models.huggingface_models.inference_model import HFInferenceModel

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

Storing as: {args.sampler.storage_path}/{args.sampler.dataset_version}_gen_{args.model_generate.name}_eval_{args.model_eval.name}_run_{args.sampler.run_id}""")


    # GET GENERATE MODEL 
    model_generate = HFInferenceModel(**args.model_generate.model_config)
    args.model_generate.completion_config.num_return_sequences = args.sampler.num_return_sequences 
    logging.info(f"Generate Model is {args.model_generate.name} on Device {model_generate.model.device}")

    # GET EVAL MODEL
    model_eval = HFInferenceModel(**args.model_eval.model_config)
    logging.info(f"Eval Model is {args.model_eval.name} on Device {model_eval.model.device}")


    # GET DATA
    data = load_dataset(**args.data.dataset)
    dataset = data[args.data.split]
    logging.info(f"Dataset is {args.data.dataset.path}. Version: {args.sampler.dataset_version}")
    
    
    # INITIALIZE CONSTITUTIONS
    constitutions = initialize_constitutions(args.sampler)
    
    
    # SAMPLE TRAINING EXAMPLES 
    all_train_examples = torch.empty(
        (
            args.sampler.constitution_batch_size, 
            args.sampler.n_revisions, 
            args.sampler.generation_batch_size,
        ),
        dtype=torch.int,
    )
    available_examples = np.arange(len(dataset))
    
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
                model=model_generate,
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


        # EVALUATION ON CURRENT DATA
        log_probs_chosen_train, log_probs_rejected_train = run_eval( 
            args=args, 
            model=model_eval,
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
            model=model_eval,
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
            model=model_eval,
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
            model=model_eval,
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
            "train_new": (log_probs_chosen_train - log_probs_rejected_train).sum(dim=-1),
            "prev_new": (log_probs_chosen_prev - log_probs_rejected_prev).sum(dim=-1),
            "train_old": (log_probs_chosen_train_old - log_probs_rejected_train_old).sum(dim=-1),
            "prev_old": (log_probs_chosen_prev_old - log_probs_rejected_prev_old).sum(dim=-1),
        }
        
                
        # UPDATE CONSTITUTIONS
        for idx in range(len(constitutions["constitutions"])):
            
            best_new_index = torch.argmax( # get best new cons across new data and prev eval data
                torch.mean(
                    torch.stack(
                        [performances["train_new"][idx].to(model_eval.model.device),
                         performances["prev_new"][idx].to(model_eval.model.device)],
                    ),
                    dim=0,
                ),
            )
                
            best_train_new = performances["train_new"][idx][best_new_index]
            best_prev_new = performances["prev_new"][idx][best_new_index] 
            best_train_old = performances["train_old"][idx]
            best_prev_old = performances["prev_old"][idx] 

            if best_train_new > best_train_old and best_prev_new > best_prev_old: 
                selected_new_constitution = new_constitutions[idx][best_new_index]
                logging.info(f"New Constitution {idx}: {selected_new_constitution}")
                constitutions["constitutions"][idx].append(selected_new_constitution)
                constitutions["log_probs_train"][idx].append(float(best_train_new))
                constitutions["log_probs_prev"][idx].append(float(best_prev_new))
            
            else:
                constitutions["constitutions"][idx].append(constitutions["constitutions"][idx][-1])
                constitutions["log_probs_train"][idx].append(float(best_train_old))
                constitutions["log_probs_prev"][idx].append(float(best_prev_old))

            constitutions["train_examples"][idx] += train_examples[idx].flatten().tolist()
            constitutions["prev_examples"][idx] += random_examples[idx].flatten().tolist()


        # WRITE TO DISK
        logging.info(f"Writing to disk.")
        constitution_ds = Dataset.from_pandas(pd.DataFrame(constitutions))
        constitution_ds.save_to_disk(f"{args.sampler.storage_path}/{args.sampler.dataset_version}_gen_{args.model_generate.name}_eval_{args.model_eval.name}_run_{args.sampler.run_id}")
  
  
if __name__ == '__main__':
    fire.Fire(main())