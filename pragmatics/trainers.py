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
from loss_functions import *


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
        average_logprobs: The log probabilities of the labels. Shape: (batch_size, ).
        logprobs
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
    
    # average token logprobs
    average_token_logprobs = per_token_logprobs.sum(dim=-1) / torch.logical_not(loss_mask).sum(dim=1)
  
    return average_token_logprobs


def kl_divergence(
    policy_logits: torch.FloatTensor,
    ref_logits: torch.FloatTensor,
) -> torch.FloatTensor:
    """Compute the kl divergence between policy and reference model.
    
    Args:
        policy_logits: Shape: (batch_size, sequence_length, vocab_size)
        ref_logits: Shape: (batch_size, sequence_length, vocab_size)        
    """
    kl = (
        policy_logits.softmax(dim=-1) * (
            policy_logits.log_softmax(dim=-1) - ref_logits.log_softmax(dim=-1))
        ).sum(dim=-1)
    
    return kl.mean()


def prepare_logits_labels(
    model: torch.nn.Module,
    tokenizer: transformers.PreTrainedTokenizer,
    batch: Dict[str, torch.Tensor],
    # model_type: str,
    ignore_idx: Optional[int] = -100,
) -> Tuple[torch.FloatTensor, torch.LongTensor]:
    """Get the logits and labels for the given prompts and responses. Includes those w/ and w/o constitutions.
    
    Args:
        model: A torch.nn.Module model.
        tokenizer: A transformers.PreTrainedTokenizer.
        batch: A batch of tokenized examples. Each value should be a tensor of shape (batch_size, sequence_length).
        model_type: policy or reference 
        ignore_idx: The index to ignore in the loss computation; defaults to -100.
        
    Returns:
        logits: The logits of the model. Shape: (batch_size, sequence_length, vocab_size).
        labels: The labels of the model. Shape: (batch_size, sequence_length).
    """
    # if model_type == 'constitution':
    prompt_attention_mask = torch.cat(
        [
            batch[key] for key in batch.keys() 
            if "prompt" in key and 'attention_mask' in key and not 'no_constitution' in key
        ],
        dim=0
    )
    
    response_attention_mask = torch.cat(
        [
            batch[key] for key in batch.keys() 
            if "response" in key and 'attention_mask' in key and not 'no_constitution' in key
        ],
        dim=0
    )
    
    responses = torch.cat(
        [
            batch[key] for key in batch.keys() 
            if "response" in key and 'attention_mask' not in key and not 'no_constitution' in key
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

    # elif model_type == 'no_constitution':
    #     prompt_attention_mask = torch.cat(
    #         [
    #             batch[key] for key in batch.keys() 
    #             if "prompt" in key and 'attention_mask' in key and 'no_constitution' in key
    #         ],
    #         dim=0
    #     )
        
    #     response_attention_mask = torch.cat(
    #         [
    #             batch[key] for key in batch.keys() 
    #             if "response" in key and 'attention_mask' in key and 'no_constitution' in key
    #         ],
    #         dim=0
    #     )
        
    #     responses = torch.cat(
    #         [
    #             batch[key] for key in batch.keys() 
    #             if "response" in key and 'attention_mask' not in key and 'no_constitution' in key
    #         ], 
    #         dim=0
    #     )

    #     labels = responses.clone()
            
    #     mask = _get_mask(
    #         attention_mask=prompt_attention_mask,
    #         labels=labels, 
    #         pad_token_id=tokenizer.pad_token_id,
    #     )

    #     labels[mask] = ignore_idx

    #     logits = model(
    #         input_ids=responses,
    #         attention_mask=response_attention_mask,
    #     ).logits

    #     return logits, labels
        


class FSDPTrainer:
    def __init__(
        self, 
        model: transformers.PreTrainedModel,
        reference_model: transformers.PreTrainedModel,
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
        self.reference_model = reference_model
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
        print('Loaded reference model on rank', self.local_rank)
        print(f"Writing checkpoints to {self.config.training.checkpoint_dir} after each epoch.")
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
        

    def compute_metrics(
        self,
        model: transformers.PreTrainedModel,
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Compute metrics."""
         # get logits and labels
        logits, labels = prepare_logits_labels(model, self.tokenizer, batch)
        logits = logits.to(model.device)
      
        # get batch logprobs
        batch_logprobs = _get_batch_logprobs(logits, labels)
        
        # reshape 
        batch_logprobs = batch_logprobs.view(self.config.data.n_constitutions,  self.config.data.n_constitutions)
        
        # compute loss
        probs, loss = pragmatic_loss_no_labels_no_reference(logprobs=batch_logprobs, max_iter=self.config.ppo.max_iter)
        
        return loss, probs, batch_logprobs  
    
    
    def compute_metrics_with_labels(
        self,
        model: transformers.PreTrainedModel,
        reference_model: transformers.PreTrainedModel,
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Compute metrics for both policy and reference model if labels are provided."""
    
        # get logprobs for policy model 
        logits, labels = prepare_logits_labels(model, self.tokenizer, batch)
        
        batch_logprobs = _get_batch_logprobs(logits, labels)
        
        batch_logprobs = batch_logprobs.view(self.config.data.n_constitutions,  self.config.data.n_constitutions)
        
        # get logprobs for reference model 
        with torch.no_grad():
            
            ref_logits, ref_labels = prepare_logits_labels(reference_model, self.tokenizer, batch)
            
            # if self.config.ppo.loss == 'reference':
            #     ref_batch_logprobs = _get_batch_logprobs(ref_logits, ref_labels)
        
        # if self.config.ppo.loss == 'reference':
            
        #     loss = pragmatic_loss_with_reference(
        #         logprobs_policy=batch_logprobs,
        #         logprobs_reference=ref_batch_logprobs,
        #     )

        # self.config.ppo.loss == 'no_reference':
            
        loss = pragmatic_loss_no_reference(logprobs=batch_logprobs)
                                              
        return loss, batch_logprobs, logits, ref_logits
        
    
    def _run_batch(
        self, 
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[torch.FloatTensor, Dict[str, torch.Tensor], torch.FloatTensor]:
        """Run batch."""    
        
        # compute policy and reference metrics
        loss, batch_logprobs, logits, ref_logits  = self.compute_metrics_with_labels(self.model, self.reference_model, batch)
            
        kl_div = kl_divergence(
            policy_logits=logits,
            ref_logits=ref_logits,
        )
      
        # loss = loss * beta * kl
        adjusted_loss = loss + self.config.ppo.beta * kl_div

        return loss, adjusted_loss, batch_logprobs, kl_div
        
            
    def evaluate(self):
        """Evaluate the model."""
        self.model.eval()
        total_loss = 0.0
        total_raw_loss = 0.0
        total_kl_div = 0.0
        n_batches = 0

        # eval
        with torch.no_grad():
            for batch in self.eval_dataloader:
                batch = {k: v.to(self.local_rank) for k, v in batch.items()}
                raw_loss, loss,  _, kl_div = self._run_batch(batch)
                total_loss += loss.item() 
                total_raw_loss += raw_loss.item() 
                total_kl_div += kl_div.item() 
                n_batches += 1
                
        # logging 
        mean_loss = total_loss / n_batches
        mean_loss = torch.tensor([mean_loss], device=self.local_rank)
        dist.all_reduce(mean_loss, op=dist.ReduceOp.SUM)
        reduced_loss = mean_loss / dist.get_world_size()
        
        mean_raw_loss = total_raw_loss / n_batches
        mean_raw_loss = torch.tensor([mean_raw_loss], device=self.local_rank)
        dist.all_reduce(mean_raw_loss, op=dist.ReduceOp.SUM)
        reduced_raw_loss = mean_raw_loss / dist.get_world_size()

        mean_kl_div = total_kl_div / n_batches
        mean_kl_div = torch.tensor([mean_kl_div], device=self.local_rank)
        dist.all_reduce(mean_kl_div, op=dist.ReduceOp.SUM)
        reduced_kl_div = mean_kl_div / dist.get_world_size()

        if self.local_rank == 0: 
            print(f"eval/loss: {reduced_loss.item()}")
            wandb.log({"eval-loss/loss": reduced_loss.item()})
            wandb.log({"eval-loss/raw-loss": reduced_raw_loss.item()})
            wandb.log({"eval-loss/kld": reduced_kl_div.item()})

    def train(self):
        """Train the model."""
        self.reference_model.eval()
        
        if self.config.training.evaluate_before_training:
            self.evaluate()
        
        for epoch in range(self.config.training.n_epochs):
            self.model.train()
            
            for step, batch in tqdm.tqdm(enumerate(self.train_dataloader), desc=f"Running epoch: {epoch}"):
                batch = {k: v.to(self.local_rank) for k, v in batch.items()}
                
                # train
                self.optimizer.zero_grad()
                raw_loss, loss, batch_logprobs, kl_div = self._run_batch(batch)
   
                (loss / self.config.training.gradient_accumulation_steps).backward()
                
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
                
                raw_loss_value = raw_loss.item() 
                raw_loss_tensor = torch.tensor([raw_loss_value], device=self.local_rank)
                dist.all_reduce(raw_loss_tensor, op=dist.ReduceOp.SUM)
                raw_reduced_loss = raw_loss_tensor / dist.get_world_size()

                dist.all_reduce(batch_logprobs, op=dist.ReduceOp.SUM)
                reduced_batch_logprobs = batch_logprobs / dist.get_world_size()
                
                dist.all_reduce(kl_div , op=dist.ReduceOp.SUM)
                reduced_kl_div = kl_div / dist.get_world_size()
                
    
                p_c0_r0 = reduced_batch_logprobs[0, 0]
                p_c0_r1 = reduced_batch_logprobs[0, 1]
                p_c1_r0 = reduced_batch_logprobs[1, 0]
                p_c1_r1 = reduced_batch_logprobs[1, 1]

                accuracy_col = ((p_c0_r0 > p_c1_r0).int() + (p_c1_r1 > p_c0_r1).int()) / 2

                accuracy_row = ((p_c0_r0 > p_c0_r1).int() + (p_c1_r1 > p_c1_r0).int()) / 2

                if self.local_rank == 0:
                    print(f"Epoch {epoch}, Step {step}: train/loss = {reduced_loss.item()}, train/raw-loss = {raw_reduced_loss.item()}, train/logprobs = {reduced_batch_logprobs}, KL = {reduced_kl_div.item()}")
                    wandb.log({"train-loss/loss": reduced_loss.item()})
                    wandb.log({"train-loss/raw-loss": raw_reduced_loss.item()})
                    wandb.log({"train-loss/kld": reduced_kl_div.item()})
                    wandb.log({"train-probs/accuracy-col": accuracy_col.item()})
                    wandb.log({"train-probs/accuracy-row": accuracy_row.item()})
                    wandb.log({"train-probs/p_c0_r0": p_c0_r0.item()})
                    wandb.log({"train-probs/p_c0_r1": p_c0_r1.item()})
                    wandb.log({"train-probs/p_c1_r0": p_c1_r0.item()})
                    wandb.log({"train-probs/p_c1_r1": p_c1_r1.item()})
                
                # evaluate after n steps have been made
                if (step + 1) % self.config.training.save_after_n_steps == 0:
                    self.evaluate()
                    self.save_checkpoint(round(step / len(self.train_dataloader), 2))
                                
            # evaluate at end of each epoch and save checkpoint 
            self.evaluate()
            self.save_checkpoint(epoch)
