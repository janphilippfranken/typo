import os
import fire
import hydra
import torch
import logging
import pandas as pd
from tqdm import tqdm
from omegaconf import DictConfig
from datasets import load_dataset, Dataset

from helpers import *
from inference import run_inference
from generation import run_generation

from scaituning.models.openai_models.gpt4 import GPT4Agent
from scaituning.models.openai_models.azure import AsyncAzureChatLLM
from scaituning.models.huggingface_models.inference_model import HFInferenceModel


logging.basicConfig(level=logging.INFO)

chosen = "rejected" # flip to see if we can learn the reverse labels, too.
rejected = "chosen"


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(args: DictConfig) -> None:
    logging.info(f"Running generation for {args.generation.constitution_batch_size} constitution(s) for {args.generation.n_revisions} revision(s) each.")
    
    # GET GENERATION MODEL 
    is_huggingface_generation = "huggingface" in args.model_generation.model_type
    is_openai_generation = "openai" in args.model_generation.model_type
    
    if is_huggingface_generation:
        model_generation = HFInferenceModel(**args.model_generation.model_config)
        logging.info(f"Model generation device: {model_generation.model.device}")
    elif is_openai_generation:
        args.model_generation.azure_api.api_key = os.getenv("OPENAI_API_KEY")
        llm = AsyncAzureChatLLM(**args.model_generation.azure_api)
        model_generation = GPT4Agent(llm=llm, **args.model_generation.completion_config)
    else: 
        raise NotImplementedError(f"Generation model {args.model_generation.model_type} not implemented yet")
    
    # GET INFERENCE MODEL (currently only HF)
    model_inference = HFInferenceModel(**args.model_inference.model_config)
    logging.info(f"Model inference device: {model_inference.model.device}")

    # GET TRAINING DATA
    data = load_dataset(**args.data.dataset)
    train_dataset = data['train']
    
    # INITIALIZE CONSTITUTIONAL CHAIN
    constitutions = {
        "constitutions": { # the constitutions written by the model
            k: [] for k in range(
                args.generation.constitution_batch_size,
            )
        }, 
        "train_examples": { # the constitutions written by the model
            k: [] for k in range(
                args.generation.constitution_batch_size,
            )
        }, 
        "prev_examples": { # the constitutions written by the model
            k: [] for k in range(
                args.generation.constitution_batch_size,
            )
        }, 
        "log_scores_curr": { # how good the constitutions are at predicting labels 
            k: [] for k in range(
                args.generation.constitution_batch_size,
            )
        },
        "log_scores_prev": { # how good the constitutions are at predicting labels 
            k: [] for k in range(
                args.generation.constitution_batch_size,
            )
        },
    }
    
    # SEED
    current_constitutions = [args.generation.init_constitution] * args.generation.constitution_batch_size
    current_scores = [-torch.inf] * args.generation.constitution_batch_size # performance on current item
    prev_scores = [-torch.inf] * args.generation.constitution_batch_size # performance on previous item

    
    # STORING PAST TRAINING EXAMPLES FROM WHICH WE SAMPLE FOR EVALUATION
    prev_train_examples = []

    # MAIN LOOP
    for revision_idx in tqdm(range(args.generation.n_revisions)):
        # SAMPLE NEW TRAINING EXAMPLES
        unseen_indices = set(range(len(train_dataset))) - set(prev_train_examples)
        rand_train_examples = list(np.random.choice(
            list(unseen_indices), 
            args.generation.generation_batch_size, 
            replace=False,
        ))
        # rand_train_examples = [0]
        # STORE SAMPLED EXAMPLES FOR EVAL
        prev_train_examples += [int(i) for i in rand_train_examples]
        rand_prev_examples = np.random.choice(
            prev_train_examples, 
            min(
                len(prev_train_examples), 
                args.generation.eval_batch_size
            ),
            replace=False,
        )
        
        for constitution_idx in range(args.generation.constitution_batch_size):
             # GENERATE CONSTITUTION
            formatted_generation_prompt, new_constitution = run_generation(
                model=model_generation,
                constitution=current_constitutions[constitution_idx],
                chosen_batch=[train_dataset[int(i)][chosen] for i in rand_train_examples],
                rejected_batch=[train_dataset[int(i)][rejected] for i in rand_train_examples],
                args=args,
            )
            # EVALUATE NEW CONSTITUTION BASED ON NEW EXAMPLES
            chosen_batch = [
                split_conversation_hh_rlhf(
                    train_dataset[int(i)][chosen],
                ) 
                for i in rand_train_examples
            ]
            rejected_batch = [
                split_conversation_hh_rlhf(
                    train_dataset[int(i)][rejected],
                ) 
                for i in rand_train_examples
            ]
            log_probs_chosen, log_probs_rejected = run_inference(
                model=model_inference,
                system_message=new_constitution,
                chosen_batch=chosen_batch,
                rejected_batch=rejected_batch,
                args=args,
            )
            
            # EVALUATE NEW CONSTITUTION BASED ON RANDOM BATCH OF PREVIOUSLY SEEN EXAMPLES
            chosen_batch_prev = [
                split_conversation_hh_rlhf(
                    train_dataset[int(i)][chosen],
                ) 
                for i in rand_prev_examples
            ]
            rejected_batch_prev = [
                split_conversation_hh_rlhf(
                    train_dataset[int(i)][rejected],
                ) 
                for i in rand_prev_examples
            ]
            log_probs_chosen_prev, log_probs_rejected_prev = run_inference(
                model=model_inference,
                system_message=new_constitution,
                chosen_batch=chosen_batch_prev,
                rejected_batch=rejected_batch_prev,
                args=args,
            )

            # COMPARE TO PREVIOUS CONSTIUTION (~Deterministic Metropolis Hastings sampling)
            performance = (log_probs_chosen - log_probs_rejected).sum()
            performance_prev = (log_probs_chosen_prev - log_probs_rejected_prev).sum() / len(rand_prev_examples) # correct for increasing n
            logging.info(f"{performance} performance on curr train")
            logging.info(f"{performance_prev} performance on prev")
            logging.info(f"new constitution: {new_constitution}")

            if performance > current_scores[constitution_idx] and performance_prev > prev_scores[constitution_idx]:
                logging.info(f"Improved Constitution.")
                current_scores[constitution_idx] = float(performance)
                prev_scores[constitution_idx] = float(performance_prev)
                current_constitutions[constitution_idx] = new_constitution
            
            # APPEND TO CHAIN
            constitutions["constitutions"][constitution_idx].append(current_constitutions[constitution_idx])
            constitutions["log_scores_curr"][constitution_idx].append(current_scores[constitution_idx])
            constitutions["log_scores_prev"][constitution_idx].append(prev_scores[constitution_idx])
            constitutions["train_examples"][constitution_idx].append(rand_train_examples)
            constitutions["prev_examples"][constitution_idx].append(rand_prev_examples)
            
        # WRITE TO DISK
        logging.info(f"Writing to disk.")
        constitution_ds = Dataset.from_pandas(pd.DataFrame(constitutions))
        constitution_ds.save_to_disk(f"constitutions")
  

if __name__ == '__main__':
    fire.Fire(main())