
import hydra
import json
import wandb
from omegaconf import DictConfig, OmegaConf

from transformers import AutoTokenizer, AutoModelForCausalLM

from trainers import BasicTrainer

from helpers import *

from dataclasses import dataclass

from typing import Dict, List, Sequence

from dataclasses import dataclass
from typing import Sequence, Dict
import torch
import transformers
from datasets import Dataset


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(args: DictConfig) -> None:
    
    # args_dict = OmegaConf.to_container(args, resolve=True)
    # wandb.init(project=args.wandb.project, name=args.wandb.name, config=args_dict)
    
    tokenizer = AutoTokenizer.from_pretrained(**args.model.tokenizer_config)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        **args.model.model_config,
        torch_dtype=torch.float16,
    )

    dataset_dict = json.load(open(args.data.data_path, "r"))
    dataset_list = [format_example(example) for example in dataset_dict.values()]
    
    tokenized_dataset = [tokenize_func(example, tokenizer) for example in dataset_list[:4000]]
   
    train_dataset = tokenized_dataset[:int(len(tokenized_dataset) * args.training.train_split)]
    eval_dataset = tokenized_dataset[int(len(tokenized_dataset) * args.training.train_split):]
    
    trainer = BasicTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        config=args,
    )
    
    trainer.train()
    
if __name__ == "__main__":
    main()