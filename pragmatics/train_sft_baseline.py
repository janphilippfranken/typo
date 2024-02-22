import os
import logging 
from tqdm import tqdm

import fire
import hydra
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from datasets import load_dataset, concatenate_datasets
from transformers import TrainingArguments, Trainer, AutoModelForCausalLM, AutoTokenizer

from helpers import *

logging.basicConfig(level=logging.INFO)

PROMPT = """System: Write a response for the assistant that completes the human request.

{conversation}

Assistant:"""


@hydra.main(version_base=None, config_path="conf", config_name="config_sft")
def main(args: DictConfig) -> None:
    logging.info(f"Writing checkpoints to: {args.training_args.output_dir}")
    logging.info(f"Max seq length: {args.model.tokenizer_config.model_max_length}")
    logging.info(f"Devices: {torch.cuda.device_count()}")
    
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
    )
    
    # training args
    training_args_dict = OmegaConf.to_container(args.training_args, resolve=True)
    training_args = TrainingArguments(**training_args_dict)
    
    # data; only take first half as we need second to generate data
    dataset_helpful = load_dataset(**args.data, data_dir='helpful-base', split='train')
    dataset_helpful = dataset_helpful.select(range(len(dataset_helpful)//2))
    dataset_harmless = load_dataset(**args.data, data_dir='harmless-base', split='train')
    dataset_harmless = dataset_harmless.select(range(len(dataset_harmless)//2))
    
    # preprocess to train on both chosen and rejected responses
    dataset = []
    for example_helpful in dataset_helpful:
        dataset.append(example_helpful)
    for example_harmless in dataset_harmless:
        dataset.append(example_harmless)

    chosen = [remove_final_answer(example['chosen']) for example in dataset]
    rejected = [remove_final_answer(example['rejected']) for example in dataset]
    examples = chosen + rejected
    
    prompts = [PROMPT.format(conversation=example[0].strip()) for example in examples]
    responses = [example[1].strip() for example in examples]
 
    dataset = sft_preprocess(prompts, responses, tokenizer)
    dataset = dataset.shuffle(42)
    logging.info(dataset)
    dataset = dataset.train_test_split(test_size=args.validation_split_size)
   
    # collator for sft
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
    
    # save trainer
    trainer.save_model(output_dir=training_args.output_dir)
    
    # save pretrained 
    output_dir = os.path.join(training_args.output_dir, "final_checkpoint")
    trainer.model.save_pretrained(output_dir)
    
if __name__ == "__main__":
    fire.Fire(main())