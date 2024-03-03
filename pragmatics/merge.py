import os
import json
import logging
import functools

import torch
import torch.distributed as dist

import hydra
from omegaconf import DictConfig, OmegaConf

from transformers import AutoTokenizer, AutoModelForCausalLM


def main():
    
    # get model
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path="mistralai/Mistral-7B-v0.1",
        cache_dir="/scr/jphilipp/scai/pretrained_models/Mistral-7B-v0.1",
    )

    # get tokenizer    
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path="mistralai/Mistral-7B-v0.1",
        cache_dir="/scr/jphilipp/scai/pretrained_models/Mistral-7B-v0.1",
        model_max_length=2048,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
                              
    # load state dict
    state_dict = torch.load('/scr/jphilipp/scai/trained_models/Mistral-7B-v0.1/checkpoints-no-icl/ppo-beta-0.1-iteration-1-0-1k/epoch-0/model.pt', map_location='cpu')
    model.load_state_dict(state_dict['state'])
        
    breakpoint()

    # /scr/jphilipp/scai/trained_models/Mistral-7B-v0.1/merged-no-icl/ppo-beta-0.1-iteration-1-0-1k/epoch-0
if __name__ == "__main__":
    main()
    
    # /scr/jphilipp/scai/trained_models/Mistral-7B-v0.1/merged-no-icl/ppo-beta-0.5-helpful-iteration-1-0-10k/epoch-0.32/model.pt