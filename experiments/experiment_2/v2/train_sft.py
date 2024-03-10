import os
import logging 
from tqdm import tqdm

import fire
import hydra
import torch
import json
import wandb
from omegaconf import DictConfig, OmegaConf
from datasets import Dataset
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
    dataset_dict_helpful = json.load(open(os.path.join(args.data.data_path, args.data.helpful)))
    dataset_dict_harmless = json.load(open(os.path.join(args.data.data_path, args.data.harmless)))
    dataset_dict_helpful_both_negative = json.load(open(os.path.join(args.data.data_path, args.data.helpful_both_negative)))
    dataset_dict_harmless_both_negative = json.load(open(os.path.join(args.data.data_path, args.data.harmless_both_negative)))
    dataset_list_helpful = [format_example_dpo(example) for example in dataset_dict_helpful.values()][:args.data.n_examples]
    dataset_list_harmless = [format_example_dpo(example) for example in dataset_dict_harmless.values()][:args.data.n_examples]
    dataset_list_helpful_both_negative = [format_example_dpo(example) for example in dataset_dict_helpful_both_negative.values()][:args.data.n_examples]
    dataset_list_harmless_both_negative = [format_example_dpo(example) for example in dataset_dict_harmless_both_negative.values()][:args.data.n_examples]

    # get sft format 
    prompts = []
    responses = []

    # add helpful examples
    for example_helpful, example_helpful_both_negative in zip(dataset_list_helpful, dataset_list_helpful_both_negative):
        prompts += example_helpful["prompts"]
        responses += example_helpful["chosen"]

        prompts += example_helpful["prompts"]
        responses += example_helpful["rejected"]
        
        prompts += example_helpful_both_negative["prompts"]
        responses += example_helpful_both_negative["chosen"]

        prompts += example_helpful_both_negative["prompts"]
        responses += example_helpful_both_negative["rejected"]

    # add harmless examples
    for example_harmless, example_harmless_both_negative in zip(dataset_list_harmless, dataset_list_harmless_both_negative):
        prompts += example_harmless["prompts"]
        responses += example_harmless["chosen"]

        prompts += example_harmless["prompts"]
        responses += example_harmless["rejected"]
        
        prompts += example_harmless_both_negative["prompts"]
        responses += example_harmless_both_negative["chosen"]

        prompts += example_harmless_both_negative["prompts"]
        responses += example_harmless_both_negative["rejected"]
    
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