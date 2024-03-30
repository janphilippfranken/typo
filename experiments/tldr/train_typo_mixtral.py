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
from transformers.models.mixtral.modeling_mixtral import MixtralDecoderLayer
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

def get_policy(blocks={MixtralDecoderLayer}):
    """Wrapping policy setup."""
    return functools.partial(
        transformer_auto_wrap_policy, transformer_layer_cls=blocks
    )

def init_distributed(
    rank: int,
    world_size: int,
    master_addr: str = "localhost",
    port: int = 12355,
    backend: str = "nccl",
):
    print(rank, "initializing distributed")
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def worker_main(rank: int, world_size: int, args: DictConfig, model):
    init_distributed(rank, world_size)
    if rank == 0:
        print("Initialized process group...")

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(**args.model.tokenizer_config)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # fsdp 
    mp_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )
    print(f"wrapping model {rank}...")
    
    model = FSDP(
        model,
        auto_wrap_policy=get_policy(),
        mixed_precision=mp_policy,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_id=torch.cuda.current_device(),
        limit_all_gathers=False,
    )
    
    print(f"wrapped model {rank}...")

    # data 
    dataset_dict = json.load(open(os.path.join(args.data_path, args.data_file)))
    dataset_list = [
        format_example(example) for example in dataset_dict.values()
    ][:args.n_examples]
    if rank == 0:
        print(f"n examples: {len(dataset_list)}")

    train_dataset = [tokenize_func(example, tokenizer) for example in dataset_list]
    random.shuffle(train_dataset)
    if rank == 0:
        print(len(train_dataset))
        print(f"tokenized {len(train_dataset)} training examples...")
    
    # train
    trainer = TYPOTrainer(
        model=model,
        reference_model=None,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=[],
        config=args,
        local_rank=rank,
        world_size=world_size,
    )
    
    trainer.train()

# main training script 
@hydra.main(version_base=None, config_path="conf", config_name="train_typo_mixtral")
def main(args: DictConfig) -> None:
    
    # nans 
    torch.autograd.set_detect_anomaly(True)

    # wandb
    args_dict = OmegaConf.to_container(args, resolve=True)
    if args.wandb.log:
        wandb.init(project=args.wandb.project, name=args.wandb.name, config=args_dict)


    # seeds
    torch.cuda.manual_seed(args.training.seed)
    torch.manual_seed(args.training.seed)
    random.seed(args.training.seed)

    # get model
    model = AutoModelForCausalLM.from_pretrained(**args.model.model_config, torch_dtype=torch.float16)

    # run with resource limits
    world_size = torch.cuda.device_count()
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
    print(f'setting RLIMIT_NOFILE soft limit to {hard} from {soft}')
    mp.spawn(worker_main, nprocs=world_size, args=(world_size, args, model))

if __name__ == "__main__":
    main()