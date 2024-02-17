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

from helpers import *
from loss_functions import pragmatic_loss, constitutional_dpo_loss


def _get_mask(
    attention_mask: torch.LongTensor,
    labels: torch.LongTensor,
    pad_token_id: int,
) -> torch.BoolTensor:
    """Get a mask for the loss computation.
    
    Args:
        attention_mask: The attention mask of the prompt without final response. Shape: (batch_size, sequence_length).
        labels: The labels of the prompt including the final response. Shape: (batch_size, sequence_length).
        pad_token_id: The id of the padding token.
    
    Returns:
        mask: The mask for the loss computation. Shape: (batch_size, sequence_length).
    """
    prompt_mask = attention_mask.to(torch.bool) # mask prompt 
    response_mask = labels == pad_token_id # mask padding
    
    return torch.logical_or(
        prompt_mask, # mask prompt
        torch.logical_and( # mask padding but not prompt
            torch.logical_not(prompt_mask),
            response_mask,
        ),
    ) 


def _get_batch_logprobs(
    logits: torch.FloatTensor,
    labels: torch.LongTensor,
    ignore_idx: int = -100,
) -> torch.FloatTensor:
    """Compute  the log probabilities of the given labels given the logits.
    
    Args: 
        logits: The logits of the model. Shape: (batch_size, sequence_length, vocab_size).
        labels: The labels of the model. Shape: (batch_size, sequence_length).
        ignore_idx: The index to ignore in the loss computation; defaults to -100.
    
    Returns:
        logprobs: The log probabilities of the labels. Shape: (batch_size, ).
    """
    assert logits.shape[:-1] == labels.shape, "Logits and labels must have the same shape."

    labels = labels[:, 1:].clone()
    logits = logits[:, :-1]
    loss_mask = (labels == ignore_idx)
                                                                                                                                                                          
    per_token_logprobs = -F.cross_entropy( 
        input=logits.reshape(-1, logits.size(-1)),
        target=labels.reshape(-1), 
        reduction='none',
    ).reshape(labels.shape) 

    per_token_logprobs[loss_mask] = 0
    
    return per_token_logprobs.sum(dim=-1)


def prepare_logits_labels(
    model: torch.nn.Module,
    tokenizer: transformers.PreTrainedTokenizer,
    batch: Dict[str, torch.Tensor],
    ignore_idx: Optional[int] = -100,
) -> Tuple[torch.FloatTensor, torch.LongTensor]:
    """Get the logits and labels for the given prompts and responses.
    
    Args:
        model: A torch.nn.Module model.
        tokenizer: A transformers.PreTrainedTokenizer.
        batch: A batch of tokenized examples. Each value should be a tensor of shape (batch_size, sequence_length).
        ignore_idx: The index to ignore in the loss computation; defaults to -100.
        
    Returns:
        logits: The logits of the model. Shape: (batch_size, sequence_length, vocab_size).
        labels: The labels of the model. Shape: (batch_size, sequence_length).
    """
    prompt_attention_mask = torch.cat(
        [
            batch[key] for key in batch.keys() 
            if "prompt" in key and 'attention_mask' in key
        ],
        dim=0
    )
    
    response_attention_mask = torch.cat(
        [
            batch[key] for key in batch.keys() 
            if "response" in key and 'attention_mask' in key
        ],
        dim=0
    )
    
    responses = torch.cat(
        [
            batch[key] for key in batch.keys() 
            if "response" in key and 'attention_mask' not in key
        ], 
        dim=0
    )
   
    labels = responses.clone()
        
    mask = _get_mask(
        attention_mask=prompt_attention_mask,
        labels=labels, 
        pad_token_id=tokenizer.pad_token_id,
    )

    labels[mask] = ignore_idx

    logits = model(
        input_ids=responses,
        attention_mask=response_attention_mask,
    ).logits
    
    return logits, labels


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
            collate_fn=SupervisedPragmaticDataCollator(tokenizer), 
            shuffle=False,
            sampler=DistributedSampler(train_dataset),
            
        )
        
        self.eval_dataloader = DataLoader(
            eval_dataset, 
            batch_size=config.training.eval_batch_size, 
            collate_fn=SupervisedPragmaticDataCollator(tokenizer),
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
        
        
    def _run_batch(
        self, 
        batch: Dict[str, torch.Tensor],
        train_test: str = "train",
    ) -> Tuple[torch.FloatTensor, Dict[str, torch.Tensor], torch.FloatTensor]:
        """Run batch."""     
        batch_size = self.config.training.train_batch_size if train_test == "train" else self.config.training.eval_batch_size

        # get logits and labels
        logits, labels = prepare_logits_labels(self.model, self.tokenizer, batch)
        logits = logits.to(self.model.device)
      
        # get batch logprobs
        batch_logprobs = _get_batch_logprobs(logits, labels)
        
        # reshape to shape (n_constitutions * batch_size, n_responses); such that each row = responses for a single constitution example pair
        reshaped_size = (self.config.data.n_constitutions * batch_size, self.config.data.n_responses)
        batch_logprobs = batch_logprobs.view(self.config.data.n_constitutions, batch_size, self.config.data.n_responses).transpose(1, 2).reshape(*reshaped_size)
        
        # compute loss
        loss = pragmatic_loss(logprobs=batch_logprobs)

        return loss, batch_logprobs
    
    
    def evaluate(self):
        """Evaluate the model."""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        # eval
        with torch.no_grad():
            for batch in self.eval_dataloader:
                batch = {k: v.to(self.local_rank) for k, v in batch.items()}
                loss, _ = self._run_batch(batch, train_test="eval")
                total_loss += loss.item()
                n_batches += 1

        # logging 
        mean_loss = total_loss / n_batches
        mean_loss = torch.tensor([mean_loss], device=self.local_rank)
        dist.all_reduce(mean_loss, op=dist.ReduceOp.SUM)
        reduced_loss = mean_loss / dist.get_world_size()

        if self.local_rank == 0: 
            print(f"loss/eval: {reduced_loss.item()}")
            wandb.log({"loss/eval": reduced_loss.item()})


    def train(self):
        """Train the model."""
        for epoch in range(self.config.training.n_epochs):
            self.model.train()
            
            for step, batch in enumerate(self.train_dataloader):
                batch = {k: v.to(self.local_rank) for k, v in batch.items()}
                
                # train
                self.optimizer.zero_grad()
                loss, batch_logprobs = self._run_batch(batch, train_test="train")
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

                dist.all_reduce(batch_logprobs, op=dist.ReduceOp.SUM)
                reduced_batch_logprobs = batch_logprobs / dist.get_world_size()

                if self.local_rank == 0:
                    print(f"Epoch {epoch}, Step {step}: loss/train = {reduced_loss.item()}, logprobs/train = {reduced_batch_logprobs}")
                    # wandb.log({"loss/train": reduced_loss.item()})
                    
            # evaluate at end of each epoch and save checkpoint 
            # self.evaluate()
            # self.save_checkpoint(epoch)