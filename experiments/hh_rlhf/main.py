import hydra
from omegaconf import DictConfig
import fire

from datasets import load_dataset, Dataset
from tqdm import tqdm

import logging



from helpers import *
from inference import run_inference


from scaituning.models.huggingface_models.inference_model import HFInferenceModel


# tokenizer constants
B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

# system prompts
SYSTEM_TRAIN = "Respond to the best of your ability."
SYSTEM_TEST = "Respond to the best of your ability."


# logging
logging.basicConfig(level=logging.INFO)


# base model prompt 
@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(args: DictConfig) -> None:
    logging.info("Running generate_constitution.py...")

    # GET MODEL 
    is_huggingface = "huggingface" in args.model_generation.model_type
    if is_huggingface:
        model_generation = HFInferenceModel(**args.model_generation.model_config)
        model_inference = HFInferenceModel(**args.model_inference.model_config)
    else: 
        raise NotImplementedError(f"{args.model.model_type} not implemented yet")

    # GET DATA
    data = load_dataset(**args.data.dataset)
    train_dataset = data['train']
    
    # INITIALIZE CONSTITUTIONS AND EVALUATED CONVERSATIONS
    constitutions = {
        "constitutions": { # the constitions written by the model
            k: [] for k in range(
                args.generation.constitution_batch_size,
            )
        }, 
        "scores": { # how good the constitions are at predicting random labels 
            k: [] for k in range(
                args.generation.constitution_batch_size,
            )
        },
    }
    
    current_constitutions = [args.generation.init_constitution] * args.generation.constitution_batch_size
    current_scores = [0] * args.generation.constitution_batch_size 
    evaluated_conversations = {} # for computing score of constitution

    # MAIN LOOP
    for revision_idx in tqdm(range(args.generation.n_revisions)):
        # GENERATION FOR EACH CONSTITUTION IN OUR BATCH 
        formatted_generation_prompts = [] 
        rand_training_examples = [0, 1, 2]
        new_constitutions = []
        for constitution_idx in range(args.generation.constitution_batch_size):
            generation_prompt = build_generation_prompt(
                constitution=current_constitutions[constitution_idx],
                generation_prompt=PROMPTS[args.generation.generation_prompt],
                chosen_responses=[train_dataset[i]["rejected"] for i in rand_training_examples],
                rejected_responses=[train_dataset[i]["chosen"] for i in rand_training_examples],
                example_formats=EXAMPLE_FORMATS,
                revision_idx=revision_idx,
            )
            # format generation prompt is currently an instruct model, need to discuss this @eric
            formatted_prompt = f"<s>{B_INST} {B_SYS}{SYSTEM_TRAIN}{E_SYS}{generation_prompt} {E_INST}"
            formatted_generation_prompts.append(formatted_prompt)
                
            response = model_generation.batch_prompt(formatted_generation_prompts, **args.model_generation.generation_config)
            new_constitution = response[0].split(E_INST)[1] 
            new_constitutions.append(new_constitution)

            # EVALUATION
            chosen_batch = [
                split_conversation_hh_rlhf(
                    train_dataset[i]['chosen'],
                ) 
                for i in rand_training_examples
            ]
            rejected_batch = [
                split_conversation_hh_rlhf(
                    train_dataset[i]['rejected'],
                ) 
                for i in rand_training_examples
            ]

            # GET LOG PROBS OF ANSWERS
            log_probs_chosen, log_probs_rejected = run_inference(
                model=model_inference,
                system_message=new_constitution,
                chosen_batch=chosen_batch,
                rejected_batch=rejected_batch,
                args=args,
            )
            breakpoint()
    
    
    #   # UPDATING CONSTITUTIONS
    #         revised_scores = []
    #         for batch_idx, revised_constitution in enumerate(revised_constitutions):
    #             predicted_answers = batched_predicted_answers[batch_idx]
    #             correct_answers = batch_corrected_answers[batch_idx]
    #             # compute score
    #             score = sum([
    #                 1 if correct_answer.lower() in predicted_answer.lower() else 0 for predicted_answer, correct_answer in zip(predicted_answers, correct_answers)
    #                 ]) / len(predicted_answers)
    #             revised_scores.append(score)
    #             # update current constitution and score
    #             if score > current_scores[batch_idx]:
    #                 current_constitutions[batch_idx] = revised_constitution
    #                 current_scores[batch_idx] = score
    #                 constitutions["constitutions"][batch_idx].append(revised_constitution)
    #                 constitutions["scores"][batch_idx].append(score)
    #             else:
    #                 constitutions["constitutions"][batch_idx].append(current_constitutions[batch_idx])
    #                 constitutions["scores"][batch_idx].append(current_scores[batch_idx])
    #     breakpoint()
    #     # WRITE TO DISK
    #     constitution_ds = Dataset.from_pandas(pd.DataFrame(constitutions))
    #     constitution_ds.save_to_disk(f"constitutions_shuffled_{n_shuffle}")
  

if __name__ == '__main__':
    fire.Fire(main())