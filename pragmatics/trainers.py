from typing import (
    Dict, 
    List, 
    Optional, 
    Sequence, 
    Tuple,
)
import numpy as np
import wandb
import os 

import matplotlib.pyplot as plt

from omegaconf import DictConfig

import torch
from torch.optim import AdamW
import torch.nn.functional as F
from torch.utils.data import DataLoader

from accelerate import Accelerator



import transformers
from transformers import get_scheduler

from datasets import Dataset

from helpers import *


N_CONST = 2
N_RESP = 2

def constitutional_dpo_loss(
    logprobs: torch.FloatTensor,
    preference_labels: torch.LongTensor,
) -> torch.FloatTensor:
    """Compute the custom DPO loss for a batch of response log probabilities.
    
    Args:
        logprobs: The log probabilities of the responses. Shape: (prompt_batch_size, response_batch_size).
        preference_labels: The true preference labels for the responses. Shape: (prompt_batch_size, response_batch_size).
        
    Returns:
        loss: The constitutional dpo loss for the batch of responses. 
    """
    prompt_batch_size = logprobs.size(0)
    chosen_cutoff = torch.tensor(prompt_batch_size // 2)

    chosen_logprobs = logprobs[:chosen_cutoff, :]
    rejected_logprobs = logprobs[chosen_cutoff:, :]
    preference_labels = preference_labels

    # compute loss for chosen responses 
    chosen_loss = -torch.sum(
        chosen_logprobs * preference_labels[:chosen_cutoff, :] -
        chosen_logprobs * preference_labels[chosen_cutoff:, :] # subtracting the logprobs of the rejected responses
    ) / chosen_cutoff

    rejected_loss = -torch.sum(
        rejected_logprobs * preference_labels[chosen_cutoff:, :] -
        rejected_logprobs * preference_labels[:chosen_cutoff, :] # subtracting the logprobs of the chosen responses
    ) / chosen_cutoff

    loss = chosen_loss + rejected_loss

    return loss


def pragmatic_loss(
    logprobs: torch.FloatTensor, 
    max_iter: int = 100,
    epsilon: float = 1e-5,
) -> torch.FloatTensor:
    """Compute the pragmatic loss for a batch of response log probabilities.
    
    Args:
        logprobs: The log probabilities of the responses. Shape: (prompt_batch_size, response_batch_size).
        max_iter: int, the maximum number of iterations for the pragmatic recursion.
        epsilon: float, the epsilon value for the loss function. As epsilon -> 0, convergence to end of pragmatic recursion.
        preference_labels: The (optional) true preference labels for the responses. Shape: (prompt_batch_size, response_batch_size).
        
    Returns:
        pragmatic_loss: The pragmatic loss for the batch of responses. Shape: (prompt_batch_size).
    """        
    probs = torch.softmax(logprobs, dim=0)  
    
    for _ in range(max_iter):
        
        # row normalization
        probs = probs / probs.sum(dim=1, keepdim=True)
        
        # check convergence
        col_sums = probs.sum(dim=0)
        if torch.max(col_sums) - torch.min(col_sums) < epsilon:
            break
        
        # column normalization
        probs = probs / probs.sum(dim=0, keepdim=True)
    
    loss = F.cross_entropy(logprobs, probs, reduction="mean")

    return loss 
    

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
    
    This function concatenates prompts and responses for tokenization, computes logits using the model, 
    and prepares labels with tokens masked out for ignoring during loss computation.
    
    Args:
        model: A torch.nn.Module model.
        tokenizer: A transformers.PreTrainedTokenizer.
        batch: A batch of tokenized examples. Each value should be a tensor of shape (batch_size, sequence_length).
        ignore_idx: The index to ignore in the loss computation; defaults to -100.
        
    Returns:
        Logits: The logits of the model. Shape: (batch_size, sequence_length, vocab_size).
        Labels: The labels of the model. Shape: (batch_size, sequence_length).
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
    
    preference_labels = torch.cat(
        [
            batch[key] for key in batch.keys() 
            if "label" in key
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
    
    return logits, labels, preference_labels


def configure_scheduler(
    optimizer: torch.optim.Optimizer,
    n_epochs: int,
    train_dataloader_length: int,
    num_warmup_steps: int = 0,
) -> torch.optim.lr_scheduler:
    """Configure and return the scheduler."""
    num_training_steps = n_epochs * train_dataloader_length
    return get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
    )
    
    
def configure_optimizer(
    model: torch.nn.Module,
    lr: float,
) -> torch.optim.Optimizer:
    """Configure and return the optimizer."""
    return AdamW(model.parameters(), lr=lr)



def _get_margins(
    logprobs: torch.FloatTensor,
) -> float:
    """Compute the margin metric for a batch of response log probabilities."""
    n = logprobs.shape[0]
    half_n = n // 2
    
    # again we assume that the first half of the batch is chosen, the second half is rejected
    first_half, second_half = logprobs[:half_n, :], logprobs[half_n:, :] 

    column_0_comparison = (first_half[:, 0] > second_half[:, 0]).float().mean()
    column_1_comparison = (second_half[:, 1] > first_half[:, 1]).float().mean()

    margin_metric = (column_0_comparison + column_1_comparison) / 2.0
    
    return margin_metric.item()


class BasicTrainer:
    def __init__(
        self, 
        model: transformers.PreTrainedModel,
        tokenizer: transformers.PreTrainedTokenizer,
        train_dataset: Dict, 
        eval_dataset: Dict, 
        config: DictConfig,
        rank: int = 0,
        world_size: int = 1,
    ):  
     
        self.rank = rank
        self.world_size = world_size
        self.model = model
        self.tokenizer = tokenizer
        
        # data loaders 
        self.train_dataloader = DataLoader(
            train_dataset, 
            batch_size=config.training.train_batch_size, 
            collate_fn=PragmaticDataCollator(tokenizer),
            shuffle=False,
        )
        
        self.eval_dataloader = DataLoader(
            eval_dataset, 
            batch_size=config.training.eval_batch_size, 
            collate_fn=PragmaticDataCollator(tokenizer),
            shuffle=False,        
        )
     
        self.config = config
        self.optimizer = configure_optimizer(model, config.training.lr)
        self.scheduler = configure_scheduler(self.optimizer, config.training.n_epochs, len(self.train_dataloader), config.training.num_warmup_steps)

        self.example_counter = 0
        self.checkpoint_dir = config.training.checkpoint_dir
     
      
        

    def compute_batch_metrics(
        self, 
        batch: Dict[str, torch.Tensor],
        batch_size: int,
        train_test: str = "train",
    ) -> Dict:
        """Compute metrics for a single batch."""
        metrics = {}
      

        logits, labels, preference_labels = prepare_logits_labels(self.model, self.tokenizer, batch)
        # reshape to N_CONST * batch_size, N_RESP
        batch_logprobs = _get_batch_logprobs(logits, labels).view(N_CONST, N_RESP, batch_size).transpose(1, 2).reshape(N_CONST * batch_size, N_RESP)
        preference_labels = preference_labels.view(N_CONST, N_RESP, batch_size).transpose(1, 2).reshape(N_CONST * batch_size, N_RESP)
        
    
        if self.config.training.loss == "constitutional_dpo":
            loss = constitutional_dpo_loss(logprobs=batch_logprobs, preference_labels=preference_labels)
        elif self.config.training.loss == "pragmatic":
            loss = pragmatic_loss(logprobs=batch_logprobs)
            
        
        margins = _get_margins(batch_logprobs)
       
        metrics[f"margins/{train_test}"] = margins
        metrics[f"loss/{train_test}"] = loss

        return loss, metrics, batch_logprobs
    

    def evaluate(self):
        """Evaluate the model."""
        self.model.eval()
        eval_losses = []

        with torch.no_grad():
            for batch in self.eval_dataloader:
                loss, metrics, _ = self.compute_batch_metrics(batch, self.config.training.eval_batch_size, train_test="eval")
                eval_losses.append(loss.item())

        avg_loss = sum(eval_losses) / len(eval_losses)
        # wandb.log({"loss/eval": avg_loss})
        # wandb.log({"margins/eval": metrics["margins/eval"]})
        
        self.model.train()

    def save_checkpoint(
        self, 
        checkpoint_name: str = "checkpoint",
    ):
        """Save model and optimizer states as a checkpoint."""
        print(f"Saving checkpoint to {self.checkpoint_dir}")
        checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_name)
        
        os.makedirs(checkpoint_path, exist_ok=True)

        self.model.save_pretrained(checkpoint_path)
        self.tokenizer.save_pretrained(checkpoint_path)
    
        torch.save({
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'config': self.config
        }, os.path.join(checkpoint_path, "training_state.bin"))

        print(f"Checkpoint saved to {checkpoint_path}")
    
    def clip_gradient(self):
        """Clip the gradient norm of the model parameters."""
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.max_grad_norm)

    def train(self):
        """Train the model."""
        
        for epoch in range(self.config.training.n_epochs):
            self.model.train()
            for batch in self.train_dataloader:
            
            
                print(f"Epoch {epoch}, Step {self.example_counter}")
            
                loss, batch_metrics, batch_logprobs = self.compute_batch_metrics(batch, self.config.training.train_batch_size, train_test="train")
                print(f"Loss: {loss.item()}, Margins: {batch_metrics['margins/train']}, Logprobs: {batch_logprobs}")
                loss.backward()
                self.clip_gradient() 
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

                # wandb.log(batch_metrics)
                # wandb.log({"learning_rate": self.optimizer.param_groups[0]['lr']})

                self.example_counter += len(batch)
                break
            
            # self.evaluate()
            # self.save_checkpoint(f"checkpoint_epoch_{epoch}")
            