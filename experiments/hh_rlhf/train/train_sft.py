import logging 
import os
import fire
import hydra
import wandb
from tqdm import tqdm
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
    
    data = [
        json.load(open(os.path.join(args.data.hh_rlhf_cai, file))) 
        for file in tqdm(os.listdir(args.data.hh_rlhf_cai),desc="Loading data...")
        if 'json' in file
    ]
    lower_threshold = 0.35
    upper_threshold = 0.65

    # Filter data
    filtered_data = [
        d for d in data 
        if lower_threshold * len(d) <= sum(i['label'] == 'chosen' for i in d) <= upper_threshold * len(d)
    ]
    data = [example for batch in filtered_data for example in batch]
    
    logging.info(f"N Train Examples: {len(data)}")  
    
    # get dpo format 
    prompts = [example['prompt'] for example in data]
    chosen = [example['chosen'] for example in data]
    
    logging.info(len(prompts))
    logging.info(len(chosen))
    logging.info(prompts[0])
    logging.info(chosen[0])

    dataset = preprocess(prompts=prompts, responses=chosen, tokenizer=tokenizer)
    dataset = dataset.shuffle(seed=42)
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
    
    output_dir = os.path.join(training_args.output_dir, "final_checkpoint")
    trainer.model.save_pretrained(output_dir)
    
if __name__ == "__main__":
    fire.Fire(main())