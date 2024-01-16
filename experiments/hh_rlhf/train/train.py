import logging 

import fire
import hydra
import wandb
import torch
from omegaconf import DictConfig, OmegaConf
from peft import LoraConfig, get_peft_model
from transformers import TrainingArguments, Trainer, AutoModelForCausalLM, AutoTokenizer

from helpers import *


logging.basicConfig(level=logging.INFO)


@hydra.main(version_base=None, config_path="conf", config_name="train")
def main(args: DictConfig) -> None:
    
    # wandb
    args_dict = OmegaConf.to_container(args, resolve=True)
    wandb.init(project=args.wandb.project, name=args.wandb.name, config=args_dict)

    # Get tokenizer 
    tokenizer = AutoTokenizer.from_pretrained(**args.model.tokenizer_config)
    
    # Set padding
    tokenizer.pad_token, tokenizer.padding_side = "[PAD]",  "right"
   
    # Get model
    model = AutoModelForCausalLM.from_pretrained(**args.model.model_config, torch_dtype=torch.float16)
    
    # Get CAI HH_RLHF Dataset 
    data_dict = jload(args.data.cai_data)  
    logging.info(f"N Train Examples: {len(data_dict)}")  
    train_dataset = preprocess(data_dict, tokenizer)
  
    # Data Collator
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    # Training Args
    training_args_dict = OmegaConf.to_container(args.training_args, resolve=True)
    training_args = TrainingArguments(**training_args_dict)
    
    ## LoRA and Peft setup
    lora_config_dict = OmegaConf.to_container(args.lora_config, resolve=True)
    lora_config = LoraConfig(**lora_config_dict)
    peft_model = get_peft_model(model, lora_config)
    logging.info(f"PEFT Params: {peft_model.print_trainable_parameters()}")  

    # Trainer
    trainer = Trainer(
        model=peft_model,
        tokenizer=tokenizer, 
        args=training_args, 
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=data_collator,
    )
    
    # Train
    trainer.train()
    
    # Save State and Model
    trainer.save_model(output_dir=training_args.output_dir)
    
    
if __name__ == "__main__":
    fire.Fire(main())