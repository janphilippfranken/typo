import fire
import hydra
import wandb
from omegaconf import DictConfig, OmegaConf
from peft import LoraConfig, get_peft_model
from transformers import TrainingArguments, Trainer

from helpers import *
from scaituning.models.huggingface_models.inference_model import HFInferenceModel


@hydra.main(version_base=None, config_path="conf", config_name="train")
def main(args: DictConfig) -> None:
    
    # wandb
    args_dict = OmegaConf.to_container(args, resolve=True)
    wandb.init(project=args.wandb.project, name=args.wandb.name, config=args_dict)

    # Get model
    model = HFInferenceModel(**args.model.model_config)
    
    # Get CAI HH_RLHF Dataset 
    data_dict = jload(args.data.cai_data)    
    train_dataset = preprocess(data_dict, model.tokenizer)
  
    # Data Collator
    data_collator = DataCollatorForSupervisedDataset(tokenizer=model.tokenizer)

    # Training Args
    training_args_dict = OmegaConf.to_container(args.training_args, resolve=True)
    training_args = TrainingArguments(**training_args_dict)
    
    ## LoRA and Peft setup
    lora_config_dict = OmegaConf.to_container(args.lora_config, resolve=True)
    lora_config = LoraConfig(**lora_config_dict)
    peft_model = get_peft_model(model.model, lora_config)
    peft_model.print_trainable_parameters()
    
    # Trainer
    trainer = Trainer(
        model=peft_model,
        tokenizer=model.tokenizer, 
        args=training_args, 
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=data_collator,
    )
    
    # Train
    trainer.train()
    
    # Save State and Model
    trainer.save_model(output_dir=training_args.output_dir)
    
    
if __name__ == "__main__":
    fire.Fire(main())