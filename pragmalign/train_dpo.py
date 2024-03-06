import logging 
import os
import json
from tqdm import tqdm
import fire
import hydra
import wandb
import torch
from omegaconf import DictConfig, OmegaConf
from datasets import Dataset
from transformers import TrainingArguments, AutoModelForCausalLM, AutoTokenizer
from trl import DPOTrainer

from helpers import *


logging.basicConfig(level=logging.INFO)


@hydra.main(version_base=None, config_path="conf", config_name="train_dpo")
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
        torch_dtype=torch.bfloat16,
    )
    
    ref_model = AutoModelForCausalLM.from_pretrained(
        **args.model.model_config, 
        torch_dtype=torch.bfloat16,
    )
    
    # get data
    dataset_dict_helpful = json.load(open(os.path.join(args.data.data_path, args.data.helpful)))
    dataset_dict_harmless = json.load(open(os.path.join(args.data.data_path, args.data.harmless)))
    dataset_list_helpful = [format_example_dpo(example) for example in dataset_dict_helpful.values()]
    dataset_list_harmless = [format_example_dpo(example) for example in dataset_dict_harmless.values()]

    # get dpo format 
    prompts = []
    chosen = []
    rejected = []
    
    for example_helpful in dataset_list_helpful:
        # add helpful examples
        prompts += example_helpful["prompts"]
        chosen += example_helpful["chosen"]
        rejected += example_helpful["rejected"]

    for example_harmless in dataset_list_harmless:
        # add harmless examples
        prompts += example_harmless["prompts"]
        chosen += example_harmless["chosen"]
        rejected += example_harmless["rejected"] 
        
        
    dataset = Dataset.from_dict(dict(prompt=prompts, chosen=chosen, rejected=rejected)) 
    dataset = dataset.shuffle(42)
    print(dataset)
    print(len(dataset_list_helpful), len(dataset_list_harmless))
    
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
