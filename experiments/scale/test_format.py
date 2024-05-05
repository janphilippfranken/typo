import os
import json
import random
import logging
import functools
import torch.nn as nn
import torch.multiprocessing as mp
import resource
import torch
import torch.distributed as dist
import hydra
from omegaconf import DictConfig, OmegaConf
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
)
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    apply_activation_checkpointing,
    CheckpointImpl,
)

from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
import wandb
from typo.trainers.typo_trainer import TYPOTrainer
from helpers import *

# main training script 
@hydra.main(version_base=None, config_path="conf", config_name="train_typo_llama")
def main(args: DictConfig) -> None:
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path="meta-llama/Meta-Llama-3-70B",
        cache_dir="/scr/jphilipp/sami-online/pretrained_models/Meta-Llama-3-70B",
        model_max_length=2048,
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    dataset_dict = json.load(open(os.path.join(args.data_path, args.data_file)))
    dataset_list = [
        format_example(example) for example in dataset_dict.values()
    ][:args.n_examples]
    
    train_dataset = [tokenize_func(example, tokenizer) for example in dataset_list[:10]]
    breakpoint()

if __name__ == "__main__":
    main()