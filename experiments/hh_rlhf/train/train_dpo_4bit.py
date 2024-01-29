import logging 
import os

import fire
import hydra
import wandb
import torch
from omegaconf import DictConfig, OmegaConf
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
from transformers import TrainingArguments, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import DPOTrainer

from helpers import *


logging.basicConfig(level=logging.INFO)


@hydra.main(version_base=None, config_path="conf", config_name="train_4bit")
def main(args: DictConfig) -> None:
    logging.info(f"Writing checkpoints to: {args.training_args.output_dir}")
    logging.info(f"Wandb name: {args.wandb.name}")
    logging.info(f"Max prompt length: {args.max_prompt_length}")
    logging.info(f"Max seq length: {args.model.tokenizer_config.model_max_length}")
    
    logging.info(args)
    
    # wandb
    args_dict = OmegaConf.to_container(args, resolve=True)
    wandb.init(project=args.wandb.project, name=args.wandb.name, config=args_dict)

    # get tokenizer
    tokenizer = AutoTokenizer.from_pretrained(**args.model.tokenizer_config)   
    tokenizer.pad_token = tokenizer.eos_token
    
    # quantization 
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_quant_type="nf4",
    )
    
    # get model 
    model = AutoModelForCausalLM.from_pretrained(
        **args.model.model_config, 
        quantization_config=quantization_config,
        device_map = "cuda",
    )
    
    # get cai dataset
    data = [
        json.load(open(os.path.join(args.data.hh_rlhf_cai, file))) 
        for file in os.listdir(args.data.hh_rlhf_cai) 
        if 'json' in file
    ]
    data = [example for batch in data for example in batch]
    logging.info(f"N Train Examples: {len(data)}")  
    
    # get dpo format 
    prompt = [example['prompt'] for example in data]
    chosen = [example['chosen'] for example in data]
    rejected = [example['rejected'] for example in data]
    example_ids = [example['example_id'] for example in data]
    filtered_prompts, filtered_chosen, filtered_rejected = filter_by_unique_ids(prompt, chosen, rejected, example_ids)
    
    # split into train/eval (just to explore loss for eval, not required)
    dataset = Dataset.from_dict(dict(prompt=filtered_prompts, chosen=filtered_chosen, rejected=filtered_rejected)) 
    dataset = dataset.shuffle(seed=42)
    logging.info(dataset)
    dataset = dataset.select(range(20000))
    dataset = dataset.train_test_split(test_size=args.validation_split_size)
    logging.info(dataset)
    logging.info(dataset['train'][0])
    
    # training args
    training_args_dict = OmegaConf.to_container(args.training_args, resolve=True)
    training_args = TrainingArguments(**training_args_dict)
    
    # LoRA and peft setup
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    
    lora_config_dict = OmegaConf.to_container(args.lora_config, resolve=True)
    lora_config = LoraConfig(**lora_config_dict)
    
    peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()
    
    logging.info(f"Devices: {torch.cuda.device_count()}")
    logging.info(f"training_args.output_dir: {training_args.output_dir}")

    # trainer
    dpo_trainer = DPOTrainer(
        model=peft_model,
        tokenizer=tokenizer,
        args=training_args, 
        beta=args.dpo_beta,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
        max_prompt_length=args.max_prompt_length,
        max_length=args.model.tokenizer_config.model_max_length,
    )
    
    # train
    dpo_trainer.train()
    dpo_trainer.save_model(output_dir=training_args.output_dir)
    
    # save final checkpoint
    output_dir = os.path.join(training_args.output_dir, "final_checkpoint")
    dpo_trainer.model.save_pretrained(output_dir)
    
    
if __name__ == "__main__":
    fire.Fire(main())