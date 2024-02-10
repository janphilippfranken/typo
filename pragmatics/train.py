
import hydra
import json
import wandb
from omegaconf import DictConfig, OmegaConf

from transformers import AutoTokenizer, AutoModelForCausalLM

from trainers import BasicTrainer


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(args: DictConfig) -> None:
    
    # args_dict = OmegaConf.to_container(args, resolve=True)
    # wandb.init(project=args.wandb.project, name=args.wandb.name, config=args_dict)
    
    tokenizer = AutoTokenizer.from_pretrained(**args.model.tokenizer_config)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(**args.model.model_config)

    dataset_dict = json.load(open(args.data.data_path, "r"))
    dataset = {int(k): v for k, v in dataset_dict.items()}
    
    train_dataset = {
        k: dataset[k] 
        for k in range(
            int(len(dataset) * args.training.train_split)
        )
    }
    
    eval_dataset = {
        k: dataset[i] 
        for k, i in enumerate(range(
            int(len(dataset) * args.training.train_split),
            len(dataset)
        ))
    }
 
    trainer = BasicTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        config=args,
    )
    
    trainer.train()
    
if __name__ == "__main__":
    main()