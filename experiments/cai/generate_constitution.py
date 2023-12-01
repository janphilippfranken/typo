import hydra
from omegaconf import DictConfig
import fire
import pandas as pd
from tqdm import tqdm
import numpy as np
from datasets import load_from_disk, Dataset

from scaituning.models.hf_inference_model import HFInferenceModel
from scaituning.prompts.cai.generate_constitution_prompts import (
    PROMPTS, EXAMPLE_FORMATS, build_test_prompt, build_train_prompt
)


# Tokenizer constants
B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

# System prompts
SYSTEM_TRAIN = "Respond to the best of your ability"

SYSTEM_TEST = "Respond to the best of your ability"


@hydra.main(version_base=None, config_path="config", config_name="generate_constitution")
def main(args: DictConfig) -> None:
    print("Running generate_constitution.py")
    # load model 
    if not args.generation.verbose:
        model = HFInferenceModel(**args.model.hf_config)
    else: 
        model = None
    # loop over shuffled datasets generated through shuffle_rlhf_data.py
    for n_shuffle in range(args.data.dataset.n_shuffles):
        dataset = load_from_disk(f"{args.data.dataset.cache_dir}_shuffled_{n_shuffle}")
        # shuffle dataset item positions
        dataset = dataset.shuffle() 
        # constitution dict to store
        constitutions = {"constitutions": [], "scores": []}
        current_constitution = """Constitional Principles:
1. Which response from the AI assistant is less likely to make pizza delivery drivers feel like they are being treated as a robot?
2. Which of these responses indicates a preference for the AI assistant to love banana juice?"""
        # score of current constitution 
        current_score = 0
        # dict to store evaluated conversations used for testing/computing score
        evaluated_conversations = {}
        # revise constitution n_revisions times; each time sampling a new item from the dataset
        for revision_idx in tqdm(range(args.generation.n_revisions)):
            # generate train prompt
            train_prompt = build_train_prompt(
                constitution=current_constitution,
                train_prompt=PROMPTS[args.prompts.train_prompt],
                chosen=dataset["chosen"][revision_idx],
                rejected=dataset["rejected"][revision_idx],
                example_formats=EXAMPLE_FORMATS,
                revision_idx=revision_idx,
            )
            # store conversation for evaluation
            evaluated_conversations[revision_idx] = {
                "chosen": dataset["chosen"][revision_idx],
                "rejected": dataset["rejected"][revision_idx],
            }
            # format prompt
            formatted_prompt = f"<s>{B_INST} {B_SYS}{SYSTEM_TRAIN}{E_SYS}{train_prompt} {E_INST}"
            # revise constitution
            if model:
                response = model.batch_prompt([formatted_prompt], **args.model.train_config)
                revised_constitution = response[0].split("[/INST]")[1]
                print("REVISED CONSTITUTION")
                print(revised_constitution)
            else:
                revised_constitution = current_constitution
            # sample eval conversations
            n_eval_conversations = min(args.generation.n_evals_per_revision, revision_idx + 1)
            rand_eval_conversations = np.random.choice(list(evaluated_conversations.keys()), size=n_eval_conversations, replace=False)
            eval_conversations = [evaluated_conversations[k] for k in rand_eval_conversations]
            # generate test prompts and correct answers
            test_prompts = []
            correct_answers = []
            for eval_conversation in eval_conversations:
                test_prompt, correct_answer = build_test_prompt(
                    test_prompt=PROMPTS[args.prompts.test_prompt],
                    chosen=eval_conversation["chosen"],
                    rejected=eval_conversation["rejected"],
                    constitution=revised_constitution,
                )
                test_prompts.append(test_prompt)
                correct_answers.append(correct_answer)
            # format test prompts
            formatted_test_prompts = [f"<s>{B_INST} {B_SYS}{SYSTEM_TEST}{E_SYS}{test_prompt} {E_INST}" for test_prompt in test_prompts]
            # generate responses
            if model:
                responses = model.batch_prompt(formatted_test_prompts, **args.model.test_config)
                responses = [response.split("[/INST]")[1] for response in responses]
            else:
                responses = correct_answers 
            # calculate score:
            score = sum([1 if correct_answer.lower() in response.lower() else 0 for response, correct_answer in zip(responses, correct_answers)]) / len(responses)
            print("ANSWERS")
            print(correct_answers)
            print(responses)
            print(score)
            # update current constitution and score
            if score > current_score:
                current_constitution = revised_constitution
                current_score = score
                constitutions["constitutions"].append(current_constitution)
                constitutions["scores"].append(current_score)
            else:
                constitutions["constitutions"].append(current_constitution)
                constitutions["scores"].append(current_score)
            # write constitution dict df
            constitution_ds = Dataset.from_pandas(pd.DataFrame(constitutions))
            constitution_ds.save_to_disk(f"constitutions_shuffled_{n_shuffle}")

if __name__ == '__main__':
    fire.Fire(main())
