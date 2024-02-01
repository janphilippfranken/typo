import os
import logging 
from tqdm import tqdm

import fire
import hydra
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from transformers import TrainingArguments, Trainer, AutoModelForCausalLM, AutoTokenizer

from helpers import *

logging.basicConfig(level=logging.INFO)


@hydra.main(version_base=None, config_path="conf", config_name="train_sft")
def main(args: DictConfig) -> None:
    logging.info(f"Writing checkpoints to: {args.training_args.output_dir}")
    logging.info(f"Wandb name: {args.wandb.name}")
    logging.info(f"Max seq length: {args.model.tokenizer_config.model_max_length}")
    logging.info(f"Devices: {torch.cuda.device_count()}")
    logging.info(f"Data: {args.data}")
    
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
    
    # data 
    if args.data.type == "human":
        prompts, chosen, rejected = get_sft_data_human(args)
        full_prompts = prompts + prompts
        full_responses = chosen + rejected
        logging.info(full_prompts[0])
        logging.info(full_prompts[len(full_prompts)//2])
        logging.info(full_responses[0])
        logging.info(full_responses[len(full_responses)//2])
    elif args.data.type == "synthetic":
        prompts, chosen, rejected = get_sft_data_synthetic(args)

    dataset = preprocess(prompts=full_prompts, responses=full_responses, tokenizer=tokenizer)
    dataset = dataset.shuffle(seed=42)

    dataset = dataset.train_test_split(test_size=args.validation_split_size)
    logging.info(dataset)
   
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