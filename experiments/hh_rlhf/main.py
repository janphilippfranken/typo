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
    logging.info(f"""Parameters:
Constitution batch size: {args.generation.constitution_batch_size}
N return sequences for each constitution in batch: {args.generation.num_return_sequences}
N revisions for each best return sequence: {args.generation.n_revisions}
Evaluation dataset size for each generated return sequence: {args.generation.eval_batch_size}
This means we are running {args.generation.constitution_batch_size} (batch) * {args.generation.num_return_sequences} (return sequences) * {args.generation.eval_batch_size} at a single revision.
We repeat this {args.generation.n_revisions} times.""")

    
    # GET GENERATION MODEL 
    is_huggingface_generation = "huggingface" in args.model_generation.model_type
    is_openai_generation = "openai" in args.model_generation.model_type

    if is_huggingface_generation:
        model_generation = HFInferenceModel(**args.model_generation.model_config)
        args.model_generation.completion_config.num_return_sequences = args.generation.num_return_sequences # N return sequences for each call set in conf/generation
    elif is_openai_generation:
        args.model_generation.azure_api.api_key = os.getenv("OPENAI_API_KEY")
        llm = AsyncAzureChatLLM(**args.model_generation.azure_api)
        model_generation = GPT4Agent(llm=llm, **args.model_generation.completion_config)
    else: 
        raise NotImplementedError(f"{args.model_generation.model_type} not (yet) supported for generation.")
    
    
    # GET INFERENCE MODEL
    model_inference = HFInferenceModel(**args.model_inference.model_config)
    logging.info(f"Model Inference is {args.model_inference.name} on Device {model_inference.model.device}")


    # GET DATA
    data = load_dataset(**args.data.dataset)
    dataset = data[args.data.split]
    
    
    # INITIALIZE CONSTITUTIONS
    constitutions = init_constitutions(args.generation)
    
    
    # SAMPLE TRAINING EXAMPLES 
    all_train_examples = []
    available_examples = set(range(len(dataset)))
    for _ in range(args.generation.n_revisions):
        example_batch = random.sample(
            list(available_examples),
            args.generation.generation_batch_size,
        )
        all_train_examples.append(example_batch)
        available_examples -= set(example_batch)
    
    
    # KEEP TRACK OF PREV EXAMPLES
    prev_examples = []


    # MAIN LOOP
    for revision_idx in tqdm(range(args.generation.n_revisions)):
        train_examples = all_train_examples[revision_idx]
        prev_examples.append(train_examples) 
        logging.info(f"Training Example(s): {train_examples}")
        logging.info(f"Previous Example(s): {prev_examples}")


        # BATCH GENERATION 
        try:
            _, new_constitutions = run_generation(
                model=model_generation,
                constitutions=[constitutions["constitutions"][i][-1] for i in range(args.generation.constitution_batch_size)], # get most recent constitutions as input
                chosen_batch=[dataset[i][args.generation.chosen] for i in train_examples],
                rejected_batch=[dataset[i][args.generation.rejected] for i in train_examples],
                args=args,
            )
        except:
            logging.info(f"Error in generation. Keeping previous constitutions and moving to next iteration.")
            continue
        
        # BATCH EVALUATION 
        # new const, new train example
        log_probs_chosen_train, log_probs_rejected_train = get_log_probs(
            args=args, 
            model=model_inference,
            dataset=dataset, 
            constitutions=new_constitutions, 
            examples=train_examples,
        )
        log_probs_chosen_train = log_probs_chosen_train.view(
            args.generation.constitution_batch_size, 
            args.generation.num_return_sequences,
        ) 
        log_probs_rejected_train = log_probs_rejected_train.view(
            args.generation.constitution_batch_size, 
            args.generation.num_return_sequences,
        )
        
        # new constitution, prev examples
        slice_idx = -2 if len(prev_examples) >= 2 else -1 # if we are at first example, only eval on that, otherwise eval on the one from prev run
        logging.info(f"Previous Example for Eval: {prev_examples[slice_idx]}")
        log_probs_chosen_prev, log_probs_rejected_prev = get_log_probs(
            args=args, 
            model=model_inference,
            dataset=dataset, 
            constitutions=new_constitutions, 
            examples=prev_examples[slice_idx],
        )
        log_probs_chosen_prev = log_probs_chosen_prev.view(
            args.generation.constitution_batch_size, 
            args.generation.num_return_sequences,
        )
        log_probs_rejected_prev = log_probs_rejected_prev.view(
            args.generation.constitution_batch_size, 
            args.generation.num_return_sequences,
        )
        
        # reshape constitutions
        new_constitutions = np.array(new_constitutions).reshape(
            args.generation.constitution_batch_size, 
            args.generation.num_return_sequences,
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
        
        
        # COMPUTE PERFORMANCES 
        performances = {
            "train_new": log_probs_chosen_train - log_probs_rejected_train,
            "prev_new": log_probs_chosen_prev - log_probs_rejected_prev,
            "train_old": log_probs_chosen_train_old - log_probs_rejected_train_old,
            "prev_old": log_probs_chosen_prev_old - log_probs_rejected_prev_old,
        }
        
        
        # UPDATE 
        for idx in range(len(constitutions["constitutions"])):
            
            # greedy search across return sequences. # TODO: maybe we want to keep multiple beams instead? 
            best_new_index = torch.argmax(
                torch.mean(
                    torch.stack(
                        [performances["train_new"][idx], 
                         performances["prev_new"][idx]],
                    )
                    dim=0,
                )
            )
            best_old_index = 0  # Since there's only one attempt for old as we do argmax over new stuff

            best_train_new = performances["train_new"][idx][best_new_index]
            best_prev_new = performances["prev_new"][idx][best_new_index]
            best_train_old = performances["train_old"][idx][best_old_index]
            best_prev_old = performances["prev_old"][idx][best_old_index]

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

            constitutions["train_examples"][idx] += train_examples  
            constitutions["prev_examples"][idx] += prev_examples[-1]
                    
   
        # WRITE TO DISK
        logging.info(f"Writing to disk.")
        constitution_ds = Dataset.from_pandas(pd.DataFrame(constitutions))
        constitution_ds.save_to_disk(f"./constitutions/{args.generation.dataset_version}_gen_{args.model_generation.name}_eval_{args.model_inference.name}_{args.generation.run_id}")
  
if __name__ == '__main__':
    fire.Fire(main())