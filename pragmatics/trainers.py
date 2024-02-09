from typing import (
    Dict, 
    List, 
    Optional, 
    Sequence, 
    Tuple,
)
import numpy as np

import matplotlib.pyplot as plt

from omegaconf import DictConfig

import torch
from torch.optim import AdamW
import torch.nn.functional as F
from torch.utils.data import DataLoader

import transformers
from transformers import get_scheduler

from datasets import Dataset

from helpers import *


def pragmatic_loss(
    logprobs: torch.FloatTensor, 
    max_iter: int = 100,
    epsilon: float = 1e-5,
    preference_labels: Optional[torch.LongTensor] = None,
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
    breakpoint()
    if preference_labels is not None:
        return F.cross_entropy(logprobs, preference_labels.float(), reduction="mean")
    
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
    ).reshape(labels.shape) # TODO: check if this is a problem for .backward() after computing pragmatic loss

    per_token_logprobs[loss_mask] = 0
    
    return per_token_logprobs.sum(dim=-1)


def prepare_logits_labels(
    model: torch.nn.Module,
    tokenizer: transformers.PreTrainedTokenizer,
    prompts: Sequence[str],
    responses: Sequence[str],
    ignore_idx: Optional[int] = -100,
) -> Tuple[torch.FloatTensor, torch.LongTensor]:
    """Get the logits and labels for the given prompts and responses.
    
    This function concatenates prompts and responses for tokenization, computes logits using the model, 
    and prepares labels with tokens masked out for ignoring during loss computation.
    
    Args:
        model: A torch.nn.Module model.
        tokenizer: A transformers.PreTrainedTokenizer.
        prompts: A list of prompt strings. Shape: (batch_size, ).
        responses: A list of response strings. Shape: (batch_size, ).
        ignore_idx: The index to ignore in the loss computation; defaults to -100.
        
    Returns:
        Logits: The logits of the model. Shape: (batch_size, sequence_length, vocab_size).
        Labels: The labels of the model. Shape: (batch_size, sequence_length).
    """
    assert len(prompts) == len(responses), "Prompts and responses must have the same length."
    
    # concatenate prompts and responses for full context
    concatenated_responses = [f"{prompt}{response}" for prompt, response in zip(prompts, responses)]
    
    tokenized_responses = tokenizer(
        concatenated_responses,
        add_special_tokens=True, 
        return_tensors="pt",
        padding=True,
    ) # tokenize concatenated responses first to get max length
     
    labels = tokenized_responses.input_ids.clone()
    
    tokenized_prompts = tokenizer(
        prompts,
        add_special_tokens=True,  
        return_tensors="pt",
        padding="max_length",
        max_length=tokenized_responses.input_ids.shape[1],  
    ) # tokenize prompts for masking
    
    logits = model(
        input_ids=tokenized_responses.input_ids,
        attention_mask=tokenized_responses.attention_mask,
    ).logits.to(torch.float32)
    
    mask = _get_mask(
        attention_mask=tokenized_prompts.attention_mask, 
        labels=labels, 
        pad_token_id=tokenizer.pad_token_id,
    ) 
   
    labels[mask] = ignore_idx

    return logits, labels


def _get_train_dataloader(
    train_dataset: Dataset,
    batch_size=1, 
    shuffle=False,
) -> torch.utils.data.DataLoader:
    """Get the train DataLoader."""
    return DataLoader(train_dataset, shuffle=shuffle, batch_size=batch_size)


def _get_eval_dataloader(
    eval_dataset: Dataset,
    batch_size=1, 
    shuffle=False,
) -> torch.utils.data.DataLoader:
    """Get the eval DataLoader."""
    return DataLoader(eval_dataset, shuffle=shuffle, batch_size=batch_size)


def configure_scheduler(
    optimizer: torch.optim.Optimizer,
    n_epochs: int,
    train_dataloader_length: int,
    num_warmup_steps=0,
) -> torch.optim.lr_scheduler:
    """Configure and return the scheduler."""
    num_training_steps = n_epochs * train_dataloader_length
    return get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
    )
    
    
def configure_optimizer(
    model: torch.nn.Module,
    lr,
) -> torch.optim.Optimizer:
    """Configure and return the optimizer."""
    return AdamW(model.parameters(), lr=lr)


def prepare_batches(
    batch_data,
) -> Tuple[List[str], List[str]]:
    """Prepare the prompt and response batches from the batch data."""
    prompt_batch = []
    response_batch = []
    preference_labels = []
    breakpoint()
    for i, examples in enumerate(batch_data): # first go over constitution dimension which has dim n const
        for j in examples['example_id']:
            for k in range(examples['n_responses'][0]):
                prompt_batch.append(examples["prompt"][j])
                response_batch.append(examples[f"r{k+1}"][j])
                preference_labels.append(1 - k) # 1 = chosen, 0 = not chosen
    preference_labels[len(preference_labels)//2:] = 1 - np.array(preference_labels[len(preference_labels)//2:]) # flip labels for anticonstitutions
    return prompt_batch, response_batch, preference_labels


class BasicTrainer:
    def __init__(
        self, 
        model: torch.nn.Module,
        tokenizer: transformers.PreTrainedTokenizer,
        train_dataset: Dataset,
        eval_dataset: Dataset,
        config: DictConfig,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
       
        self.train_dataloader = _get_train_dataloader(train_dataset, self.config.training.batch_size)
        self.eval_dataloader = _get_eval_dataloader(eval_dataset, self.config.training.batch_size)
        self.optimizer = configure_optimizer(model, self.config.training.lr)
        self.scheduler = configure_scheduler(
            self.optimizer, self.config.training.n_epochs, len(self.train_dataloader)
        )

    def train(self):
        losses = []
        for epoch in range(self.config.training.n_epochs):
            self.model.train()
            for batch in self.train_dataloader:

                prompt_batch, response_batch, preference_labels_batch = prepare_batches(batch)
                breakpoint()
                logits, labels = prepare_logits_labels(
                    self.model, self.tokenizer, prompt_batch, response_batch, 
                )
                
                batch_logprobs = _get_batch_logprobs(logits, labels)
       
                batch_logprobs = batch_logprobs.reshape(self.config.training.batch_size, self.config.data.n_responses)
                preference_labels_batch = torch.tensor(preference_labels_batch).reshape(self.config.training.batch_size * self.config.data.n_responses, -1)
                
                breakpoint()
                loss = pragmatic_loss(logprobs=batch_logprobs, preference_labels=preference_labels_batch)
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                print(f"Epoch: {epoch}, Loss: {loss.item()}")
                losses.append(loss.item())
                print(batch_logprobs)
                plot_and_save_logprobs(batch_logprobs, epoch)
                break
                # breakpoint()
        # plot loss
        plt.plot(losses)
        # save 
        plt.savefig("losses.pdf")