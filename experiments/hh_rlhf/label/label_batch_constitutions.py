import json
import logging
import hydra
from tqdm import tqdm
from datasets import load_dataset

from scaituning.models.vllm_models.inference_model import VLLMInferenceModel
from helpers import *

logging.basicConfig(level=logging.INFO)

BOS_TOKEN = "<s>"

@hydra.main(version_base=None, config_path="conf", config_name="config_batch_constitutions")
def main(args: DictConfig) -> None:
    
    # get model 
    model = VLLMInferenceModel(**args.model.model_config)

    # get training data
    dataset = load_dataset(**args.data.dataset)
    dataset_train = dataset['train']
    
    example_idx = 0

    # outer loop for constitution batches
    for batch_idx, constitution_start in enumerate(tqdm(
        range(args.label.constitution_start, args.label.constitution_start + args.label.max_constitutions, args.label.constitution_batch_size),
        desc=f"Labelling Constitutions {args.label.constitution_start} to {args.label.constitution_start + args.label.max_constitutions}",
    )):
        
        # initialize containers for storing results
        results = []
        constitution_strings = []
        constitution_ids = []

        # get constitutions
        for constitution_idx in range(constitution_start, constitution_start + args.label.constitution_batch_size):
            with open(f"{args.label.synthetic_constitutions:}/constitution_{constitution_idx}.json", 'r') as file: 
                constitution = json.load(file)

            constitution_strings.append(constitution["constitution"])
            constitution_ids.append(constitution["constitution_id"])
        
        # process and label training examples
        for constitution_string, constitution_id in zip(constitution_strings, constitution_ids):
            
            log_probs_no_constitution = get_log_probs_of_answer(
                model=model,
                dataset=dataset_train,
                constitutions=[""],
                eval_prompt=args.label.evaluation_prompt,
                example_idx=example_idx,
            )

            log_probs_constitution = get_log_probs_of_answer(
                model=model,
                dataset=dataset_train,
                constitutions=[constitution_string],
                eval_prompt=args.label.evaluation_prompt,
                example_idx=example_idx,
            )

            # compare log probabilities to get labels
            labels = (
                log_probs_constitution['batch_log_probs_chosen'] - log_probs_no_constitution["batch_log_probs_chosen"]
            ) > (
                log_probs_constitution['batch_log_probs_rejected'] - log_probs_no_constitution["batch_log_probs_rejected"]
            )
            
           
            chosen_answer = log_probs_constitution['final_answer_chosen']
            rejected_answer = log_probs_constitution['final_answer_rejected']
            log_probs_chosen = float(log_probs_constitution['batch_log_probs_chosen'][0])
            log_probs_rejected = float(log_probs_constitution['batch_log_probs_rejected'][0])
            label = int(labels[0])

            example = {
                'prompt': log_probs_constitution['prompts'][0].split(BOS_TOKEN)[1],
                'chosen': chosen_answer if label else rejected_answer,
                'rejected': rejected_answer if label else chosen_answer,
                'label': 'chosen' if label else 'rejected',
                'log_probs_chosen': log_probs_chosen,
                'log_probs_rejected': log_probs_rejected,
                'constitution_id': constitution_id,
                'example_id': example_idx,
            }
                
            results.append(example)

        # save after each batch
        filename = f"{args.label.storage_path}/constitutions-{batch_idx}-example-{example_idx}.json"
        with open(filename, "w") as f:
            json.dump(results, f)

        example_idx += 1


if __name__ == '__main__':
    main()