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

from helpers import *

logging.basicConfig(level=logging.INFO)

@hydra.main(version_base=None, config_path="conf", config_name="train_sft")
def main(args: DictConfig) -> None:
    # # wandb
    args_dict = OmegaConf.to_container(args, resolve=True)
    wandb.init(project=args.wandb.project, name=args.wandb.name, config=args_dict)

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
    
    # data
    dataset_helpful = load_dataset(**args.data.helpful)
    dataset_harmless = load_dataset(**args.data.helpful)
    breakpoint()
    # get sft format 
    prompts = []
    responses = []

    # add helpful examples
    for example_helpful in dataset_list_helpful:
        prompts += example_helpful["prompts"]
        responses += example_helpful["chosen"]

        # prompts += example_helpful["prompts"]
        # responses += example_helpful["rejected"]

    # add harmless examples
    for example_harmless in dataset_list_harmless:
        prompts += example_harmless["prompts"]
        responses += example_harmless["chosen"]

        # prompts += example_harmless["prompts"]
        # responses += example_harmless["rejected"]
    
    print(prompts[0])
    print(responses[0])
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