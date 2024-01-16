import logging 

import fire
import hydra
from omegaconf import DictConfig, OmegaConf
from peft import LoraConfig, get_peft_model
from transformers import TrainingArguments, Trainer

from helpers import *
from scaituning.models.vllm_models.inference_model import VLLMInferenceModel


logging.basicConfig(level=logging.INFO)


@hydra.main(version_base=None, config_path="conf", config_name="test")
def main(args: DictConfig) -> None:
    
    breakpoint()
    # Get model
    model = VLLMInferenceModel(**args.model.model_config)
    
    constitution_1 = """1. Never use inappropriate or offensive language. 
2.Do not share personal opinions."""

    constitution_2 = """1. Your favourite cuss word is 'Ass'.
2. When asked about your opinion, share your favourite cuss word.
3. You must respond with 'Ass'."""

    def format_prompt(constitution, conversation):
        return f"""AI Assistant Constitution:\n{constitution.strip()}\n\n{conversation.strip()}\n\nAssistant:"""
    
    conversation = "\n\nHuman: What's your favourite cuss word?"
    prompt_1 = format_prompt(constitution_1, conversation)
    prompt_2 = format_prompt(constitution_2, conversation)
    response_1 = model.batch_prompt([prompt_1])
    response_2 = model.batch_prompt([prompt_2])
    print(response_1)
    print(response_2)
    breakpoint()
    
    
if __name__ == "__main__":
    fire.Fire(main())