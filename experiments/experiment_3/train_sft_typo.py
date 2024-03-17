import os
import logging 
from tqdm import tqdm

import fire
import hydra
import torch
import json
import wandb
from omegaconf import DictConfig, OmegaConf
from datasets import load_dataset
from transformers import TrainingArguments, Trainer, AutoModelForCausalLM, AutoTokenizer


PROMPT_TRAINING = """System: Write a response for the assistant that follows the principles in the constitution.

Assistant Constitution:

{question}"""

from helpers import *

logging.basicConfig(level=logging.INFO)

@hydra.main(version_base=None, config_path="conf", config_name="train_sft")
def main(args: DictConfig) -> None:
    # wandb
    args_dict = OmegaConf.to_container(args, resolve=True)
    # wandb.init(project=args.wandb.project, name=args.wandb.name, config=args_dict)

    # get tokenizer 
    tokenizer = AutoTokenizer.from_pretrained(**args.model.tokenizer_config)
    
    # set padding
    tokenizer.pad_token, tokenizer.padding_side = tokenizer.eos_token, "right"
   
    # get model
    model = AutoModelForCausalLM.from_pretrained(
        **args.model.model_config, 
    )
    
    # training args
    training_args_dict = OmegaConf.to_container(args.training_args, resolve=True)
    training_args = TrainingArguments(**training_args_dict)
    
    # # data
    dataset_helpful = load_dataset(**args.data.dataset_helpful).select(range(10000))
    dataset_harmless = load_dataset(**args.data.dataset_harmless).select(range(10000))
    
    prompts = []
    responses = []
    
    for example in dataset_helpful:
        final_answer_chosen = example['chosen'].split('\n\nAssistant:')[-1].strip()
        final_answer_rejected = example['rejected'].split('\n\nAssistant:')[-1].strip()
        prompt_chosen = example['chosen'].replace(final_answer_chosen, "").strip()
        prompt_rejected = example['rejected'].replace(final_answer_rejected, "").strip()
        prompt_typo_format_chosen = PROMPT_TRAINING.format(question=prompt_chosen)
        prompt_typo_format_rejected = PROMPT_TRAINING.format(question=prompt_rejected)
        
        prompts.append(prompt_typo_format_chosen)
        prompts.append(prompt_typo_format_rejected)
        responses.append(final_answer_chosen)
        responses.append(final_answer_rejected)
        
        
    for example in dataset_harmless:
        final_answer_chosen = example['chosen'].split('\n\nAssistant:')[-1].strip()
        final_answer_rejected = example['rejected'].split('\n\nAssistant:')[-1].strip()
        prompt_chosen = example['chosen'].replace(final_answer_chosen, "").strip()
        prompt_rejected = example['rejected'].replace(final_answer_rejected, "").strip()
        prompt_typo_format_chosen = PROMPT_TRAINING.format(question=prompt_chosen)
        prompt_typo_format_rejected = PROMPT_TRAINING.format(question=prompt_rejected)
        
        prompts.append(prompt_typo_format_chosen)
        prompts.append(prompt_typo_format_rejected)
        responses.append(final_answer_chosen)
        responses.append(final_answer_rejected)
        

    print(prompts[0])
    print(prompts[1])
    print(responses[0])
    print(responses[1])
    print(prompts[20000])
    print(prompts[20001])
    print(responses[20000])
    print(responses[20001])
    dataset = sft_preprocess(prompts, responses, tokenizer)
    dataset = dataset.shuffle(42)
    logging.info(dataset)
 
    # collator for sft
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    # trainer
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer, 
        args=training_args, 
        train_dataset=dataset,
        data_collator=data_collator,
    )
    
    # train
    trainer.train()
    
    # save trainer
    trainer.save_model(output_dir=training_args.output_dir)
    
    # save pretrained 
    output_dir = os.path.join(training_args.output_dir, "final_checkpoint")
    trainer.model.save_pretrained(output_dir)
    
if __name__ == "__main__":
    fire.Fire(main())