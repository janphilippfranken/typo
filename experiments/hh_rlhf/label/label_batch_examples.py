import json
import logging
import hydra
from tqdm import tqdm
from datasets import load_dataset

from scaituning.models.vllm_models.inference_model import VLLMInferenceModel
from helpers import *

logging.basicConfig(level=logging.INFO)

BOS_TOKEN = "<s>"

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(args: DictConfig) -> None:
    
    # get model 
    model = VLLMInferenceModel(**args.model.model_config)

    # get training data
    dataset = load_dataset(**args.data.dataset)
    dataset_train = dataset['train']
    
    results = []

    # get constitutions
    for constitution_idx in range(args.label.constitution_batch_size):
        with open(f"{args.label.synthetic_constitutions:}/constitution_{constitution_idx}.json", 'r') as file: 
            constitution = json.load(file)

        constitution_string = constitution["constitution"]
        constitution_id = constitution["constitution_id"]
     
        log_probs_no_constitution = get_log_probs_of_answer(
            model=model,
            dataset=dataset_train,
            constitutions=[""],
            eval_prompt=args.label.evaluation_prompt,
            example_idx=constitution_idx,
        )

        log_probs_constitution = get_log_probs_of_answer(
            model=model,
            dataset=dataset_train,
            constitutions=[constitution_string],
            eval_prompt=args.label.evaluation_prompt,
            example_idx=constitution_idx,
        )

        # compare log probabilities to get labels
        labels = (
            log_probs_constitution['batch_log_probs_chosen'] - log_probs_no_constitution["batch_log_probs_chosen"]
        ) > (
            log_probs_constitution['batch_log_probs_rejected'] - log_probs_no_constitution["batch_log_probs_rejected"]
        )
        
        # construct fine-tuning examples
       
        chosen_answer = log_probs_constitution['final_answer_chosen']
        rejected_answer = log_probs_constitution['final_answer_rejected']
        log_probs_chosen = float(log_probs_constitution['batch_log_probs_chosen'][0])
        log_probs_rejected = float(log_probs_constitution['batch_log_probs_rejected'][0])
        logging.info(labels)
        label = int(labels[0])

        example = {
            'prompt': log_probs_constitution['prompts'][constitution_idx].split(BOS_TOKEN)[1],
            'chosen': chosen_answer if label else rejected_answer,
            'rejected': rejected_answer if label else chosen_answer,
            'label': 'chosen' if label else 'rejected',
            'log_probs_chosen': log_probs_chosen,
            'log_probs_rejected': log_probs_rejected,
            'constitution_id': constitution_id,
            'example_id': constitution_idx,
        }
        
        results.append(example)

        # save after each batch
        filename = f"{args.label.storage_path}/constitution-{constitution_idx}-example-{constitution_idx}.json"
        with open(filename, "w") as f:
            json.dump(results, f)


if __name__ == '__main__':
    main()