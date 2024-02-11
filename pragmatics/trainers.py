from typing import (
    Dict, 
    List, 
    Optional, 
    Tuple,
)

import os 

import wandb
from omegaconf import DictConfig

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

import transformers
from transformers import get_scheduler

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


class BasicTrainer:
    def __init__(
        self, 
        model: transformers.PreTrainedModel,
        tokenizer: transformers.PreTrainedTokenizer,
        config: DictConfig,
        train_dataset: List[Dict],
        eval_dataset: List[Dict],
    ):  
        """Intialize the trainer.
        
        Args:
            model: A transformers.PreTrainedModel.
            tokenizer: A transformers.PreTrainedTokenizer.
            train_dataset: List of training examples. 
            eval_dataset: List of evaluation examples.
            config: Training configuration.  
            rank: The rank of the process in distributed training.
            world_size: The number of processes in distributed training.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
    
        
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
     
        
        self.optimizer = AdamW(model.parameters(), lr=config.training.lr)
        self.scheduler = get_scheduler(name="linear", optimizer=self.optimizer, num_warmup_steps=config.training.num_warmup_steps, num_training_steps=config.training.num_training_steps)
    
        self.example_counter = 0
        self.checkpoint_dir = config.training.checkpoint_dir
        
        
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

     
    def _run_batch(
        self, 
        batch: Dict[str, torch.Tensor],
        train_test: str = "train",
    ) -> Tuple[torch.FloatTensor, Dict[str, torch.Tensor], torch.FloatTensor]:
        """Compute metrics for a single batch."""     
        metrics = {}
        
        batch_size = self.config.training.train_batch_size if train_test == "train" else self.config.training.eval_batch_size

        # get logits and labels
        logits, labels, preference_labels = prepare_logits_labels(self.model, self.tokenizer, batch)
      
        # get batch logprobs
        batch_logprobs = _get_batch_logprobs(logits, labels)
        
        # reshape to shape (n_constitutions * batch_size, n_responses); such that each row = responses for a single constitution example pair
        batch_logprobs = batch_logprobs.view(
            self.config.data.n_constitutions, 
            self.config.data.n_responses, 
            batch_size).transpose(1, 2).reshape(
                self.config.data.n_constitutions * batch_size, 
                self.config.data.n_responses
            )
        
        # same for preference labels; shape (n_constitutions * batch_size, n_responses)
        preference_labels = preference_labels.view(
            self.config.data.n_constitutions, 
            self.config.data.n_responses, 
            batch_size).transpose(1, 2).reshape(
                self.config.data.n_constitutions * batch_size, 
                self.config.data.n_responses
            )

        # compute loss
        if self.config.training.loss == "constitutional_dpo":
            loss = constitutional_dpo_loss(logprobs=batch_logprobs, preference_labels=preference_labels)
        
        # pragmatic loss requires no preference labels
        elif self.config.training.loss == "pragmatic":
            loss = pragmatic_loss(logprobs=batch_logprobs)
            
        # margins; currently only works for two responses
        assert self.config.data.n_responses == 2, "Margin metric currently only works for two responses per example."
        margins = get_margins(batch_logprobs)
       
        metrics[f"margins/{train_test}"] = margins
        metrics[f"loss/{train_test}"] = loss

        return loss, metrics, batch_logprobs
    

    def evaluate(self):
        """Evaluate the model."""
        self.model.eval()
        eval_losses = []

        with torch.no_grad():
            for batch in self.eval_dataloader:
                loss, metrics, _ = self._run_batch(batch, train_test="eval")
                eval_losses.append(loss.item())

        avg_loss = sum(eval_losses) / len(eval_losses)
        print(f"Eval loss: {avg_loss}, Eval margins: {metrics['margins/eval']}")
        
        wandb.log({"loss/eval": avg_loss})
        wandb.log({"margins/eval": metrics["margins/eval"]})
        

    def train(self):
        """Train the model."""
        for epoch in range(self.config.training.n_epochs):
            
            self.model.train()
            
            for batch in self.train_dataloader:
                print(f"Epoch {epoch}, Step {self.example_counter}")
                
                loss, batch_metrics, batch_logprobs = self._run_batch(batch, train_test="train")
                print(f"Loss: {loss.item()}, Margins: {batch_metrics['margins/train']}, Logprobs: {batch_logprobs}")
                
                loss.backward()
                self.clip_gradient() 
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

                wandb.log(batch_metrics)
                wandb.log({"learning_rate": self.optimizer.param_groups[0]['lr']})

                self.example_counter += len(batch)
                break
            
            self.evaluate()
            self.save_checkpoint(f"checkpoint_epoch_{epoch}")