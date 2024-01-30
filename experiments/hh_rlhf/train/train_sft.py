import logging 
import os
import fire
import hydra
import wandb
import torch
from omegaconf import DictConfig, OmegaConf
from datasets import load_dataset, Dataset
from peft import LoraConfig, get_peft_model
from transformers import TrainingArguments, Trainer, AutoModelForCausalLM, AutoTokenizer

from helpers import *


logging.basicConfig(level=logging.INFO)


PROMPT = """Write a response for the assistant that completes the human request.

{conversation}

Assistant:"""


@hydra.main(version_base=None, config_path="conf", config_name="train_sft")
def main(args: DictConfig) -> None:
    logging.info(f"Writing checkpoints to: {args.training_args.output_dir}")
    logging.info(f"Wandb name: {args.wandb.name}")
    logging.info(f"Max seq length: {args.model.tokenizer_config.model_max_length}")
    logging.info(f"Devices: {torch.cuda.device_count()}")
    
    # wandb
    args_dict = OmegaConf.to_container(args, resolve=True)
    wandb.init(project=args.wandb.project, name=args.wandb.name, config=args_dict)

    # get tokenizer 
    tokenizer = AutoTokenizer.from_pretrained(**args.model.tokenizer_config)
    
    # set padding
    tokenizer.pad_token, tokenizer.padding_side = "[PAD]", "right"
   
    # get model
    model = AutoModelForCausalLM.from_pretrained(
        **args.model.model_config, 
        torch_dtype=torch.bfloat16,
    )

    # training args
    training_args_dict = OmegaConf.to_container(args.training_args, resolve=True)
    training_args = TrainingArguments(**training_args_dict)

    
    # get helpful base dataset
    dataset = load_dataset(
        path="Anthropic/hh-rlhf", 
        data_dir="helpful-base", 
        cache_dir="/scr/jphilipp/scai/datasets/hh-rlhf",
    )["train"]

    prompts = [
        PROMPT.format(conversation=remove_final_answer(example['chosen'])[0].strip())
        for example in dataset
    ]
    logging.info(prompts[0])

    responses = [
        remove_final_answer(example['chosen'])[1].strip()
        for example in dataset
    ]
    logging.info(responses[0])

    dataset = preprocess(prompts=prompts, responses=responses, tokenizer=tokenizer)
    dataset = dataset.train_test_split(test_size=args.validation_split_size)
    logging.info(dataset)
   
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    # trainer
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer, 
        args=training_args, 
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
        data_collator=data_collator,
    )
    
    # train
    trainer.train()
    
    # save model
    trainer.save_model(output_dir=training_args.output_dir)
    
if __name__ == "__main__":
    fire.Fire(main())