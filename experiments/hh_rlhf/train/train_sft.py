import logging 

import fire
import hydra
import wandb
import torch
from omegaconf import DictConfig, OmegaConf
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import TrainingArguments, Trainer, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from helpers import *


logging.basicConfig(level=logging.INFO)


@hydra.main(version_base=None, config_path="conf", config_name="train")
def main(args: DictConfig) -> None:
    logging.info(f"Dataset is: {args.data.cai_data}")
    logging.info(f"Writing checkpoints to: {args.training_args.output_dir}")
    logging.info(f"Wandb name: {args.wandb.name}")
    
    # wandb
    args_dict = OmegaConf.to_container(args, resolve=True)
    wandb.init(project=args.wandb.project, name=args.wandb.name, config=args_dict)

    # Get tokenizer 
    tokenizer = AutoTokenizer.from_pretrained(**args.model.tokenizer_config)
    
    # Set padding
    tokenizer.pad_token, tokenizer.padding_side = "[PAD]",  "right"
   
    # Get model
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # Get model
    model = AutoModelForCausalLM.from_pretrained(
        **args.model.model_config, 
        quantization_config=quantization_config,
        device_map="cuda",
    )
    
    # Get CAI HH_RLHF Dataset 
    data_dict = jload(args.data.cai_data)  
    logging.info(f"N Train Examples: {len(data_dict)}")  
    dataset = preprocess(data_dict, tokenizer)
    dataset = dataset.train_test_split(test_size=args.validation_split_size)
    logging.info(dataset)

    # Data Collator
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    # Training Args
    training_args_dict = OmegaConf.to_container(args.training_args, resolve=True)
    training_args = TrainingArguments(**training_args_dict)
    
    ## LoRA and Peft setup
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    
    lora_config_dict = OmegaConf.to_container(args.lora_config, resolve=True)
    lora_config = LoraConfig(**lora_config_dict)
    
    peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()
    
    peft_model.is_parallelizable = True
    peft_model.model_parallel = True
    
    logging.info(f"Devices: {torch.cuda.device_count()}")

    # Trainer
    trainer = Trainer(
        model=peft_model,
        tokenizer=tokenizer, 
        args=training_args, 
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
        data_collator=data_collator,
    )
    
    # Train
    trainer.train()
    
    # Save State and Model
    trainer.save_model(output_dir=training_args.output_dir)
    
    
if __name__ == "__main__":
    fire.Fire(main())