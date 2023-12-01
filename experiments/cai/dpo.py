import hydra
from omegaconf import DictConfig

import pandas as pd
from tqdm import tqdm

from datasets import load_dataset, Dataset

from scaituning.models.hf_inference_model import HFInferenceModel

from scaituning.prompts.cai.cai_prompts import CAI_PROMPTS, EXAMPLE_FORMATS, create_shuffled_train_batch


B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
SYSTEM = "Respond to the best of your ability."


def get_stack_exchange_paired(
    sanity_check: bool = False,
    cache_dir: str = "/scr/jphilipp/scai/datasets",
    num_proc=24,
) -> Dataset:
    """Load the stack-exchange-paired dataset from Hugging Face and convert it to the necessary format.

    The dataset is converted to a dictionary with the following structure:
    {
        'prompt': List[str],
        'chosen': List[str],
        'rejected': List[str],
    }

    Prompts are structured as follows:
      "Question: " + <prompt> + "\n\nAnswer: "
    """
    dataset = load_dataset(
        "lvwerra/stack-exchange-paired",
        split="train",
        cache_dir=cache_dir,
    )
    original_columns = dataset.column_names

    if sanity_check:
        dataset = dataset.select(range(min(len(dataset), 10)))

    def return_prompt_and_responses(samples) -> Dict[str, str]:
        return {
            "prompt": ["Question: " + question + "\n\nAnswer: " for question in samples["question"]],
            "chosen": samples["response_j"],
            "rejected": samples["response_k"],
        }

    return dataset.map(
        return_prompt_and_responses,
        batched=True,
        num_proc=num_proc,
        remove_columns=original_columns,
    )




@hydra.main(version_base=None, config_path="config", config_name="config")
def main(args: DictConfig) -> None:


    # 2. Load the Stack-exchange paired dataset
    train_dataset = get_stack_exchange_paired()
    breakpoint()

    
if __name__ == '__main__':
    main()