import logging
import os 
import fire
import json
import hydra
from tqdm import tqdm
import torch
from omegaconf import DictConfig
import copy

import copy 

from typing import Dict, List, Optional, Tuple, Sequence

import transformers
from datasets import load_dataset, Dataset, load_from_disk

from transformers import DataCollatorForLanguageModeling

from helpers import *

from scaituning.models.huggingface_models.inference_model import HFInferenceModel


logging.basicConfig(level=logging.INFO)






BOS_TOKEN = "<s>"
EOS_TOKEN = "</s>"


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(args: DictConfig) -> None:
    
    # Model
    model = HFInferenceModel(**args.model.model_config)
    

    # CAI HH_RLHF Dataset 
  
    data_dict = jload(args.data.cai_data)
        

    sources = [
        f"{BOS_TOKEN}{formatting_func(example['constitution'], example['conversation'])}"
        for example in data_dict[:5]
    ]
    targets = [f"{example['final_response'].strip()}{EOS_TOKEN}" for example in data_dict[:5]]
    
    data_dict = preprocess(sources, targets, model.tokenizer)
    
    train_dataset = Dataset.from_dict(data_dict)
    train_dataset.set_format('torch')
    data_collator = DataCollatorForSupervisedDataset(tokenizer=model.tokenizer)
    breakpoint()
    
    trainer = Trainer(
        model=model.model, 
        tokenizer=model.tokenizer, 
        args=training_args, 
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=data_collator,
    )
    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)
    
if __name__ == "__main__":
    fire.Fire(main())