import logging 

import fire
import hydra
from omegaconf import DictConfig, OmegaConf
from peft import PeftModel
from transformers import TrainingArguments, Trainer

from helpers import *
from scaituning.models.huggingface_models.inference_model import HFInferenceModel


logging.basicConfig(level=logging.INFO)


@hydra.main(version_base=None, config_path="conf", config_name="merge_peft_model")
def main(args: DictConfig) -> None:
    

    # Get model
    model = HFInferenceModel(**args.model.model_config)
    tokenizer = model.tokenizer

    # breakpoint()    
    model = PeftModel.from_pretrained(
        model=model.model,
        model_id=args.peft_model_id,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    breakpoint()
    model = model.merge_and_unload()
    model.save_pretrained(args.merged_peft_model_name)

    
    
if __name__ == "__main__":
    fire.Fire(main())