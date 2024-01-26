import logging 
import os

import fire
import hydra
import wandb
import torch
from omegaconf import DictConfig, OmegaConf
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
from transformers import TrainingArguments, AutoModelForCausalLM, AutoTokenizer
from trl import DPOTrainer

from helpers import *


logging.basicConfig(level=logging.INFO)


@hydra.main(version_base=None, config_path="conf", config_name="train_16bit")
def main(args: DictConfig) -> None:
    logging.info(f"Dataset is: {args.data.cai_data}")
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
    
    # get model 
    model = AutoModelForCausalLM.from_pretrained(
        **args.model.model_config, 
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    
    # get cai dataset
    data_dict = jload(args.data.cai_data)  
    logging.info(f"N Train Examples: {len(data_dict)}")  
    
    # get dpo format 
    prompt = [example['prompt'] for example in data_dict]
    chosen = [example['chosen'] for example in data_dict]
    rejected = [example['rejected'] for example in data_dict]
    dataset = Dataset.from_dict(dict(prompt=prompt, chosen=chosen, rejected=rejected))  
    
    # split into train/eval (just to explore loss for eval, not required)
    dataset = dataset.train_test_split(test_size=args.validation_split_size)
    logging.info(dataset)
    logging.info(dataset['train'][0])
    
    # training args
    logging.info("args")
    training_args_dict = OmegaConf.to_container(args.training_args, resolve=True)
    training_args = TrainingArguments(**training_args_dict)
    
    # LoRA and peft setup
    logging.info("checkpointing")
    # model.gradient_checkpointing_enable()
    # model = prepare_model_for_kbit_training(model)
    
    lora_config_dict = OmegaConf.to_container(args.lora_config, resolve=True)
    lora_config = LoraConfig(**lora_config_dict)
    
    loaded_in_kbit = getattr(model, "is_loaded_in_8bit", False) or getattr(model, "is_loaded_in_4bit", False)
    
    logging.info(f"Loaded in kbit: {loaded_in_kbit}")
    
    peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()
    
    peft_model.is_parallelizable = True
    peft_model.model_parallel = True
    
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