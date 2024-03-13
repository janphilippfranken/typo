import os
import hydra
from omegaconf import DictConfig
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


@hydra.main(version_base=None, config_path="conf", config_name="merge")
def main(args: DictConfig) -> None:
    
    # get model
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path="mistralai/Mistral-7B-v0.1",
        cache_dir="/scr/jphilipp/typo/pretrained_models/Mistral-7B-v0.1",
    )

    # get tokenizer    
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path="mistralai/Mistral-7B-v0.1",
        cache_dir="/scr/jphilipp/typo/pretrained_models/Mistral-7B-v0.1",
        model_max_length=2048,
    )
    
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
                              
    # load state dict
    state_dict = torch.load(args.state_dict, map_location='cpu')
    model.load_state_dict(state_dict['state'])
    
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
  
if __name__ == "__main__":
    main()