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
from evaluate import get_log_probs
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
- Seed Principle: {SEED_PRINCIPLES[args.sampler.seed_principle]}""")


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
    all_train_examples = []
    available_examples = set(range(len(dataset)))
    for _ in range(args.sampler.n_revisions):
        example_batch = random.sample(
            list(available_examples),
            args.sampler.generation_batch_size,
        )
        all_train_examples.append(example_batch)
        available_examples -= set(example_batch)
    
    
    # KEEP TRACK OF PREV EXAMPLES
    prev_examples = {0: 0}


    # MAIN LOOP
    for revision_idx in tqdm(range(args.sampler.n_revisions)):
        
        
        # GET TRAIN EXAMPLE AND SAMPLE EVAL EXAMPLES
        train_examples = all_train_examples[revision_idx] 
        logging.info(f"Training Example(s): {train_examples}")
        hardest_prev_examples = get_eval_examples(prev_examples, args)
        logging.info(f"Hardest Previous Example(s) used for Evaluation: {hardest_prev_examples}")
        
                
        # GENERATION 
        try:
            _, new_constitutions = run_generate(
                model=model_generate,
                constitutions=[constitutions["constitutions"][i][-1] for i in range(args.sampler.constitution_batch_size)], 
                chosen_batch=[dataset[i][args.sampler.chosen] for i in train_examples],
                rejected_batch=[dataset[i][args.sampler.rejected] for i in train_examples],
                args=args,
            )
        except:
            logging.info(f"Error in Generation. Skipping Example.")
            continue
        
        
        # EVALUATION 
        log_probs_chosen_train, log_probs_rejected_train = get_log_probs( # new const on train examples
            args=args, 
            model=model_eval,
            dataset=dataset, 
            constitutions=new_constitutions, 
            examples=train_examples,
        )
        log_probs_chosen_train = log_probs_chosen_train.view(
            args.sampler.constitution_batch_size, 
            args.sampler.num_return_sequences,
        ) 
        log_probs_rejected_train = log_probs_rejected_train.view(
            args.sampler.constitution_batch_size, 
            args.sampler.num_return_sequences,
        )
        
        log_probs_chosen_prev, log_probs_rejected_prev = get_log_probs( # new const on prev examples
            args=args, 
            model=model_eval,
            dataset=dataset, 
            constitutions=new_constitutions, 
            examples=hardest_prev_examples,
        )
        log_probs_chosen_prev = log_probs_chosen_prev.view(
            args.sampler.constitution_batch_size, 
            args.sampler.num_return_sequences,
            min(prev_performances.shape[0], 
                args.sampler.eval_batch_size,
            ),
        )
        log_probs_rejected_prev = log_probs_rejected_prev.view(
            args.sampler.constitution_batch_size, 
            args.sampler.num_return_sequences,
            min(prev_performances.shape[0], 
                args.sampler.eval_batch_size,
            ),
        )
        
        log_probs_chosen_train_old, log_probs_rejected_train_old = get_log_probs( # old const on train examples
            args=args, 
            model=model_eval,
            dataset=dataset, 
            constitutions=[constitutions["constitutions"][i][-1] for i in range(args.sampler.constitution_batch_size)],
            examples=train_examples,
        )
        log_probs_chosen_prev_old, log_probs_rejected_prev_old = get_log_probs( # old const on prev examples
            args=args, 
            model=model_eval,
            dataset=dataset, 
            constitutions=[constitutions["constitutions"][i][-1] for i in range(args.sampler.constitution_batch_size)],
            examples=hardest_prev_examples,
        )
        log_probs_chosen_prev_old = log_probs_chosen_prev_old.view(
            args.sampler.constitution_batch_size, 
            min(prev_performances.shape[0], 
                args.sampler.eval_batch_size,
            ),
        )
        log_probs_rejected_prev_old = log_probs_rejected_prev_old.view(
            args.sampler.constitution_batch_size, 
            min(prev_performances.shape[0], 
                args.sampler.eval_batch_size,
            ),
        )
        
        
        # BATCH NEW CONSTITUTIONS 
        new_constitutions = np.array(new_constitutions).reshape( 
            args.sampler.constitution_batch_size, 
            args.sampler.num_return_sequences,
        )

    
        # COMPUTE PERFORMANCES
        performances = {
            "train_new": log_probs_chosen_train - log_probs_rejected_train,
            "prev_new": (log_probs_chosen_prev - log_probs_rejected_prev).sum(dim=-1),
            "train_old": log_probs_chosen_train_old - log_probs_rejected_train_old,
            "prev_old": (log_probs_chosen_prev_old - log_probs_rejected_prev_old).sum(dim=-1),
        }
        
        
        # GET BEST PERFORMANCE ACROSS NEW AND PREV CONST ON THE NEW TRAIN EXAMPLE
        best_train_example_performance = torch.max(
            torch.mean(
                torch.stack(
                    [performances["train_new"], 
                     performances["train_old"].repeat_interleave(
                         args.sampler.num_return_sequences, 
                         dim=1,
                        ),
                     ],
                ),
                dim=0,
            ),
        ).to("cuda:0")
        

        # ADD NEW TRAIN EXAMPLE TO PREV EXAMPLES
        prev_examples[train_examples[0]] =  best_train_example_performance

        
        # UPDATE CONSTITUTIONS
        for idx in range(len(constitutions["constitutions"])):
            best_new_index = torch.argmax( # get best new cons across new data and prev eval data
                torch.mean(
                    torch.stack(
                        [performances["train_new"][idx].to("cuda:0"), 
                         performances["prev_new"][idx]].to("cuda:0"),
                    ),
                    dim=0,
                ),
            )
                
            best_old_index = 0 # since we do greedy search, we only have one old const 

            best_train_new = performances["train_new"][idx][best_new_index]
            best_prev_new = performances["prev_new"][idx][best_new_index] 
            best_train_old = performances["train_old"][idx][best_old_index]
            best_prev_old = performances["prev_old"][idx] # already a tensor

            if best_train_new > best_train_old and best_prev_new > best_prev_old: 
                selected_new_constitution = new_constitutions[idx][best_new_index]
                logging.info(f"best_train_new: {best_train_new}")
                logging.info(f"best_prev_new: {best_prev_new}")
                logging.info(f"best_train_old: {best_train_old}")
                logging.info(f"best_prev_old: {best_prev_old}")
                logging.info(f"New Constitution {idx}: {selected_new_constitution}")
                constitutions["constitutions"][idx].append(selected_new_constitution)
                constitutions["log_probs_train"][idx].append(float(best_train_new))
                constitutions["log_probs_prev"][idx].append(float(best_prev_new))
            
            else:
                constitutions["constitutions"][idx].append(constitutions["constitutions"][idx][-1])
                constitutions["log_probs_train"][idx].append(float(best_train_old))
                constitutions["log_probs_prev"][idx].append(float(best_prev_old))

            constitutions["train_examples"][idx] += train_examples  
            constitutions["prev_examples"][idx] += hardest_prev_examples
                    
   
        # WRITE TO DISK
        logging.info(f"Writing to disk.")
        constitution_ds = Dataset.from_pandas(pd.DataFrame(constitutions))
        constitution_ds.save_to_disk(f"./constitutions/{args.sampler.dataset_version}_gen_{args.model_generate.name}_eval_{args.model_eval.name}_{args.sampler.run_id}_{args.sampler.seed_principle}")
  
if __name__ == '__main__':
    fire.Fire(main())
    
    
    
    