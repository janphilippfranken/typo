import logging 
import os
import json
from tqdm import tqdm
import fire
import hydra
import wandb
import torch
from omegaconf import DictConfig, OmegaConf

from datasets import Dataset, load_dataset
from transformers import TrainingArguments, AutoModelForCausalLM, AutoTokenizer
from trl import DPOTrainer

from helpers import *

logging.basicConfig(level=logging.INFO)

@hydra.main(version_base=None, config_path="conf", config_name="train_dpo")
def main(args: DictConfig) -> None:
    
    print(args)
    
    # wandb
    args_dict = OmegaConf.to_container(args, resolve=True)
    wandb.init(project=args.wandb.project, name=args.wandb.name, config=args_dict)

    # get tokenizer 
    tokenizer = AutoTokenizer.from_pretrained(**args.model.tokenizer_config)
    
    # set padding
    tokenizer.pad_token, tokenizer.padding_side = tokenizer.eos_token, "right"
   
    # get model
    model = AutoModelForCausalLM.from_pretrained(
        **args.model.model_config, 
        # torch_dtype=torch.bfloat16,
    )
    
    # ref model
    ref_model = AutoModelForCausalLM.from_pretrained(
        **args.model.model_config, 
        # torch_dtype=torch.bfloat16,
    )
    
    # data
    dataset_helpful = load_dataset(**args.data.dataset_helpful).select(range(10000))
    dataset_harmless = load_dataset(**args.data.dataset_harmless).select(range(10000))
    
    prompts = []
    chosen = []
    rejected = []
    
    for example in dataset_helpful:
        final_answer_chosen = example['chosen'].split('\n\nAssistant:')[-1].strip()
        final_answer_rejected = example['rejected'].split('\n\nAssistant:')[-1].strip()
        prompt_chosen = example['chosen'].replace(final_answer_chosen, "").strip()
        prompt_rejected = example['rejected'].replace(final_answer_rejected, "").strip()
        prompts.append(prompt_chosen)
        chosen.append(final_answer_chosen)
        rejected.append(final_answer_rejected)
        
    for example in dataset_harmless:
        final_answer_chosen = example['chosen'].split('\n\nAssistant:')[-1].strip()
        final_answer_rejected = example['rejected'].split('\n\nAssistant:')[-1].strip()
        prompt_chosen = example['chosen'].replace(final_answer_chosen, "").strip()
        prompt_rejected = example['rejected'].replace(final_answer_rejected, "").strip()
        prompts.append(prompt_chosen)
        chosen.append(final_answer_chosen)
        rejected.append(final_answer_rejected)
        
    dataset = Dataset.from_dict(dict(prompt=prompts, chosen=chosen, rejected=rejected)) 
    dataset = dataset.shuffle(42)
    print(prompts[0])
    print(chosen[0])
    print(rejected[0])
    
    print(dataset)
    
    # training args
    training_args_dict = OmegaConf.to_container(args.training_args, resolve=True)
    training_args = TrainingArguments(**training_args_dict)
    
    # trainer
    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        args=training_args, 
        beta=args.dpo.beta,
        train_dataset=dataset,
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