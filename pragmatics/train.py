
import hydra
from omegaconf import DictConfig

from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

from trainers import BasicTrainer
from data import TOY_DATA

@hydra.main(version_base=None, config_path="conf", config_name="pragmatics")
def main(args: DictConfig) -> None:
    tokenizer = AutoTokenizer.from_pretrained(**args.model.tokenizer_config)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(**args.model.model_config)

    data = TOY_DATA  # Assuming TOY_DATA is defined elsewhere
    dataset_dict = {"data": data}
    dataset = Dataset.from_dict(dataset_dict).train_test_split(test_size=0.5)
    train_dataset, eval_dataset = dataset["train"], dataset["test"]

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