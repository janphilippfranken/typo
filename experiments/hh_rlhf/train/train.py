import fire
import hydra
import wandb
from omegaconf import DictConfig, OmegaConf
from transformers import TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model


from helpers import *
from scaituning.models.huggingface_models.inference_model import HFInferenceModel


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(args: DictConfig) -> None:
    
    # wandb
    wandb.init(project=args.train.wandb.project, name=args.train.wandb.name) #, config=args)

    # Get model
    model = HFInferenceModel(**args.model.model_config)
    
    # Get CAI HH_RLHF Dataset 
    data_dict = jload(args.data.cai_data)    
    train_dataset = preprocess(data_dict, model.tokenizer)
  
    # Data Collator
    data_collator = DataCollatorForSupervisedDataset(tokenizer=model.tokenizer)

    # Training Args
    training_args = TrainingArguments(**args.train.training_args)
    
    ## LoRA and Peft setup
    lora_config = LoraConfig(**args.train.lora_config)
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