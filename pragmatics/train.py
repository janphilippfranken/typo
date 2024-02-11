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

from trainers import BasicTrainer
from helpers import format_example, tokenize_func


def print_model(model, file_name, local_rank):
    if local_rank != 0:
        return
    with open(file_name, "w") as external_file:
        print(f"model wrapping = \n{model}\n\n", file=external_file)

def setup():
    """Set up for torchrun."""
    dist.init_process_group("nccl")
    
def cleanup():
    dist.destroy_process_group()

def clear_gpu_cache(rank=None):
    if rank == 0:
        print(f"clearing gpu cache for all ranks")
    torch.cuda.empty_cache()

def setup_tasks(rank, world_size):
    setup()
    clear_gpu_cache(rank)
    
def get_policy(blocks={MistralDecoderLayer}):
    """Wrapping policy setup."""
    return functools.partial(
        transformer_auto_wrap_policy, transformer_layer_cls=blocks)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(args: DictConfig) -> None:
    
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ["WORLD_SIZE"])
    
    torch.cuda.manual_seed(local_rank)
    torch.manual_seed(local_rank)
    
    if rank == 0:
        assert torch.cuda.device_count() == world_size, "World size does not match CUDA device count."
    
    setup_tasks(rank, world_size)
    if torch.distributed.is_initialized():
        torch.cuda.set_device(local_rank)
        
    my_auto_wrap_policy = get_policy()
    
    tokenizer = AutoTokenizer.from_pretrained(**args.model.tokenizer_config)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        **args.model.model_config, torch_dtype=torch.float16)
        
    if local_rank == 0:
        num_params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"Model with {num_params}M params prepared")
        
    mp_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )
        
    model = FSDP(
        model,
        auto_wrap_policy=my_auto_wrap_policy,
        mixed_precision=mp_policy,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_id=torch.cuda.current_device(),
    )
    
    dataset_dict = json.load(open(args.data.data_path))
    dataset_list = [format_example(example) for example in dataset_dict.values()]
    
    tokenized_dataset = [tokenize_func(example, tokenizer) for example in dataset_list[:100]]
    split_index = int(len(tokenized_dataset) * args.training.train_split)
    train_dataset = tokenized_dataset[:split_index]
    eval_dataset = tokenized_dataset[split_index:]
    
    trainer = BasicTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        config=args,
        local_rank=local_rank,
    )
    
    trainer.train()
    cleanup()
    
if __name__ == "__main__":
    main()
