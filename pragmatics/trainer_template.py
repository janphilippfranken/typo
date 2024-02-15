from typing import (
    Dict, 
    List, 
    Optional, 
    Tuple,
)

import os 
import tqdm

import wandb
from omegaconf import DictConfig

import torch
from torch.optim import AdamW
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp.api import FullStateDictConfig, FullOptimStateDictConfig

import transformers
from transformers import get_scheduler

from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    StateDictType,
)


class FSDPTrainer:
    def __init__(
        self, 
        model: transformers.PreTrainedModel,
        tokenizer: transformers.PreTrainedTokenizer,
        config: DictConfig,
        train_dataset: List[Dict],
        eval_dataset: List[Dict],
        local_rank: int,
        world_size: int,
    ):  
        """Intialize the Trainer.
        
        Args:
            model: A transformers.PreTrainedModel.
            tokenizer: A transformers.PreTrainedTokenizer.
            train_dataset: List of training examples. 
            eval_dataset: List of evaluation examples.
            config: Training configuration.  
            local_rank: the rank for distributed training
            world_size: num gpus 
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.local_rank = local_rank
        self.world_size = world_size
        
        # data loaders 
        self.train_dataloader = DataLoader(
            train_dataset, 
            batch_size=config.training.train_batch_size, 
            collate_fn=YOURCOLLATOR(tokenizer), 
            shuffle=False,
            sampler=DistributedSampler(train_dataset),
            
        )
        
        self.eval_dataloader = DataLoader(
            eval_dataset, 
            batch_size=config.training.eval_batch_size, 
            collate_fn=YOURCOLLATOR(tokenizer),
            shuffle=False,
            sampler=DistributedSampler(eval_dataset),
        )
        
        self.optimizer = AdamW(model.parameters(), lr=config.training.lr)
        
        self.scheduler = get_scheduler(
            name="linear", 
            optimizer=self.optimizer, 
            num_warmup_steps=config.training.num_warmup_steps, 
            num_training_steps=(
                len(self.train_dataloader) * self.config.training.n_epochs
                ) // config.training.gradient_accumulation_steps,
        )
    
        self.checkpoint_dir = config.training.checkpoint_dir
        print('Loaded model on rank', self.local_rank)
        dist.barrier()


    def write_state_dict(
        self, 
        epoch: int, 
        state: Dict[str, torch.Tensor], 
        checkpoint_name: str,
    ):
        """Write a checkpoint to disk."""
        checkpoint_dir = os.path.join(self.config.training.checkpoint_dir, f"epoch-{epoch}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        file_path = os.path.join(checkpoint_dir, checkpoint_name)
        torch.save({
            'epoch': epoch,
            'state': state,
        }, file_path)
    
    
    def save_checkpoint(self, epoch):
        """Save model, optimizer, and scheduler state to disk, gathering from all processes and saving only on the rank 0 process."""
        # model
        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT, state_dict_config=save_policy):
            model_state_dict = self.model.state_dict()
        
        if self.local_rank == 0:
            self.write_state_dict(
                epoch=epoch,
                state=model_state_dict, 
                checkpoint_name='model.pt'
            )
        del model_state_dict
        dist.barrier()

        # optimizer 
        save_policy = FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT, optim_state_dict_config=save_policy):
            optimizer_state_dict = FSDP.optim_state_dict(self.model, self.optimizer)

        if self.local_rank == 0:
            self.write_state_dict(
                epoch=epoch,
                state=optimizer_state_dict, 
                checkpoint_name='optimizer.pt',
            )
        del optimizer_state_dict
        dist.barrier()

        # scheduler
        if self.local_rank == 0:
            scheduler_state_dict = self.scheduler.state_dict()
            self.write_state_dict(
                epoch=epoch,
                state=scheduler_state_dict,
                checkpoint_name='scheduler.pt',
            )
        dist.barrier()
    
     
    def _run_batch(self, batch: Dict[str, torch.Tensor], train_test: str) -> torch.FloatTensor:
        """Run batch."""     
        loss = 0
        return loss
    
    def evaluate(self):
        """Evaluate the model."""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        # eval
        with torch.no_grad():
            for batch in self.eval_dataloader:
                batch = {k: v.to(self.local_rank) for k, v in batch.items()}
                loss = self._run_batch(batch, train_test="eval")
                total_loss += loss.item()
                n_batches += 1

        # logging 
        mean_loss = total_loss / n_batches
        mean_loss = torch.tensor([mean_loss], device=self.local_rank)
        dist.all_reduce(mean_loss, op=dist.ReduceOp.SUM)
        reduced_loss = mean_loss / dist.get_world_size()

        if self.local_rank == 0: 
            print(f"loss/eval: {reduced_loss.item() / self.config.training.gradient_accumulation_steps}")
            wandb.log({"loss/eval": reduced_loss.item() / self.config.training.gradient_accumulation_steps})


    def train(self):
        """Train the model."""
        for epoch in range(self.config.training.n_epochs):
            self.model.train()
            
            for step, batch in enumerate(self.train_dataloader):
                batch = {k: v.to(self.local_rank) for k, v in batch.items()}
                
                # train
                self.optimizer.zero_grad()
                loss = self._run_batch(batch, train_test="train")
                loss = loss / self.config.training.gradient_accumulation_steps
                loss.backward()
                
                # accumulate
                if (step + 1) % self.config.training.gradient_accumulation_steps == 0 or (step + 1) == len(self.train_dataloader):
                    self.model.clip_grad_norm_(self.config.training.max_grad_norm)
                    self.optimizer.step()
                    self.scheduler.step()

                # logging
                loss_value = loss.item()
                loss_tensor = torch.tensor([loss_value], device=self.local_rank)
                dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
                reduced_loss = loss_tensor / dist.get_world_size()

                if self.local_rank == 0:
                    print(f"Epoch {epoch}, Step {step}: loss/train = {reduced_loss.item()}")
                    wandb.log({"loss/train": reduced_loss.item()})
                    
            # evaluate at end of each epoch and save checkpoint 
            self.evaluate()
            self.save_checkpoint(epoch)