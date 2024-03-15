import os
import json
import random
import logging
import functools

import torch
import torch.distributed as dist

import hydra
from omegaconf import DictConfig, OmegaConf

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.models.mistral.modeling_mistral import MistralDecoderLayer 

from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

import wandb

from typo.trainers.typo_trainer import TYPOTrainer
from helpers import *

# ddp/fsdp utils for launching with torchrun
def setup():
    dist.init_process_group("nccl")
    
def cleanup():
    dist.destroy_process_group()

def clear_gpu_cache(rank=None):
    if rank == 0:
        print(f"clearing gpu cache for all ranks")
    torch.cuda.empty_cache()

def setup_tasks(rank):
    setup()
    clear_gpu_cache(rank)

def get_policy(blocks={MistralDecoderLayer}):
    """Wrapping policy setup."""
    return functools.partial(
        transformer_auto_wrap_policy, 
        transformer_layer_cls=blocks,
    )

# main training script 
@hydra.main(version_base=None, config_path="conf", config_name="train_typo")
def main(args: DictConfig) -> None:
    # for nans
    torch.autograd.set_detect_anomaly(True)
    print(args.training)
    
    # wandb
    args_dict = OmegaConf.to_container(args, resolve=True)
    wandb.init(project=args.wandb.project, name=args.wandb.name, config=args_dict)
    
    # distributed setup
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    
    if local_rank == 0:
        assert torch.cuda.device_count() == world_size, "World size does not match CUDA device count."
        logging.info(f"beta: {args.typo.beta}")
        logging.info(f"writing checkpoints to: {args.training.checkpoint_dir}")
    
    setup_tasks(local_rank)
    
    if torch.distributed.is_initialized():
        torch.cuda.set_device(local_rank)
         
    # seeds
    torch.cuda.manual_seed(args.training.seed)
    torch.manual_seed(args.training.seed)
    random.seed(args.training.seed)
    
    # get tokenizer    
    tokenizer = AutoTokenizer.from_pretrained(**args.model.tokenizer_config)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # get model 
    model = AutoModelForCausalLM.from_pretrained(
        **args.model.model_config)
    
    reference_model = AutoModelForCausalLM.from_pretrained(
        **args.model.model_config)
    
    # load archived .pt file 
    if args.training.model_archive:
        state_dict = torch.load(args.training.model_archive, map_location='cpu')
        model.load_state_dict(state_dict['state'])
    
    if local_rank == 0:
        print('Base model:', args.model.model_config)
        num_params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"Model with {num_params}M params prepared")  
    
    # fsdp setup 
    mp_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )
    
    # wrap model 
    model = FSDP(
        model,
        auto_wrap_policy=get_policy(), # currently hardcoded for mistraldecoderlayer
        mixed_precision=mp_policy,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_id=torch.cuda.current_device(),
        limit_all_gathers=False,
    )
    
    reference_model = FSDP(
        reference_model,
        auto_wrap_policy=get_policy(), 
        mixed_precision=mp_policy,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_id=torch.cuda.current_device(),
        limit_all_gathers=False,
    )
    
    # get data
    print(os.path.join(args.data_path, args.helpful))
    print(os.path.join(args.data_path, args.harmless))
    dataset_dict_helpful = json.load(open(os.path.join(args.data_path, args.helpful)))
    dataset_dict_harmless = json.load(open(os.path.join(args.data_path, args.harmless)))
    dataset_list_helpful = [format_example(example) for example in dataset_dict_helpful.values()][:args.n_examples]
    dataset_list_harmless = [format_example(example) for example in dataset_dict_harmless.values()][:args.n_examples]

    if local_rank == 0:
        print(f"n helpful: {len(dataset_list_helpful)}")
        print(f"n harmless: {len(dataset_list_harmless)}")
        print(dataset_list_helpful[0])
        print(dataset_list_harmless[0])
    
   
    tokenized_dataset_helpful = [tokenize_func(example, tokenizer) for example in dataset_list_helpful]
    tokenized_dataset_harmless = [tokenize_func(example, tokenizer) for example in dataset_list_harmless]

    train_dataset = tokenized_dataset_helpful + tokenized_dataset_harmless 
    
    random.shuffle(train_dataset)
   
    if local_rank == 0:
        print(len(train_dataset))
        print(f"tokenized {len(train_dataset)} training examples...")

    # train
    trainer = TYPOTrainer(
        model=model,
        reference_model=reference_model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=[],
        config=args,
        local_rank=local_rank,
        world_size=world_size,
    )
    
    trainer.train()
    cleanup()
    
if __name__ == "__main__":
    main()