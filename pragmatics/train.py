import os
import json
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

from trainers import FSDPTrainer
from helpers import format_model_written_example, tokenize_func_no_label

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
@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(args: DictConfig) -> None:
    
    # wandb
    # args_dict = OmegaConf.to_container(args, resolve=True)
    # wandb.init(project=args.wandb.project, name=args.wandb.name, config=args_dict)
    
    # distributed setup
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    
    if local_rank == 0:
        assert torch.cuda.device_count() == world_size, "World size does not match CUDA device count."
    
    setup_tasks(local_rank)
    
    if torch.distributed.is_initialized():
        torch.cuda.set_device(local_rank)
         
    # seeds
    torch.cuda.manual_seed(args.training.seed)
    torch.manual_seed(args.training.seed)
    
    # # get tokenizer    
    tokenizer = AutoTokenizer.from_pretrained(**args.model.tokenizer_config)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # get model 
    model = AutoModelForCausalLM.from_pretrained(
        **args.model.model_config)
    
    # load archived .pt file 
    if args.training.model_archive:
        state_dict = torch.load(args.training.model_archive, map_location='cpu')
        model.load_state_dict(state_dict['state'])
    
  
    if local_rank == 0:
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
    
    # get data
    dataset_dict = json.load(open(args.data.data_path))
    dataset_list = [format_model_written_example(example) for example in dataset_dict.values()]

    tokenized_dataset = [tokenize_func_no_label(example, tokenizer) for example in dataset_list[:4]]

    train_dataset = tokenized_dataset[:int(len(tokenized_dataset) * args.training.train_split)]
    eval_dataset = tokenized_dataset[int(len(tokenized_dataset) * args.training.train_split):]

    # train
    trainer = FSDPTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        config=args,
        local_rank=local_rank,
        world_size=world_size,
    )
    
    trainer.train()
    cleanup()
    
if __name__ == "__main__":
    main()