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


def typo_loss(
    logprobs: torch.FloatTensor,
) -> torch.FloatTensor:
    """Computes the typo loss.
    
    Args:
        logprobs: Log probabilities of the responses from the policy. Shape: (n_constitutions, n_responses).

    Returns:
        typo_loss: Typo loss.
    """
    # col-wise normalization constant
    logsumexp = torch.logsumexp(logprobs, dim=0, keepdim=True)
    
    # logits
    logits = logprobs - logsumexp
    logits_t = logits.t()
    
    # labels = identity matrix
    labels = torch.arange(logprobs.size(0)).long().to(logprobs.device)
    
    # row- and col-wise cross entropy 
    loss_row = F.cross_entropy(logits, labels, reduction="mean")
    loss_col = F.cross_entropy(logits_t, labels, reduction="mean") 
    
    return (loss_row + loss_col) / 2


def _get_mask(
    attention_mask: torch.LongTensor,
    labels: torch.LongTensor,
    pad_token_id: int,
) -> torch.BoolTensor:
    """Returns a mask for the loss computation.
    
    Args:
        attention_mask: The attention mask of the prompt without final response. Shape: (batch_size, sequence_length).
        labels: The labels of the prompt including the final response. Shape: (batch_size, sequence_length).
        pad_token_id: The id of the padding token.
    
    Returns:
        mask: The mask for the loss computation. Shape: (batch_size, sequence_length).
    """
    prompt_mask = attention_mask.to(torch.bool) # mask prompt 
    response_mask = labels == pad_token_id      # mask padding
    
    return torch.logical_or(
        prompt_mask,       # mask prompt
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
    """Computes the log probabilities of labels given logits.
    
    Args: 
        logits: The logits of the model. Shape: (batch_size, sequence_length, vocab_size).
        labels: The labels of the model. Shape: (batch_size, sequence_length).
        ignore_idx: The index to ignore in the loss computation; defaults to -100.
    
    Returns:
        average_logprobs: The log probabilities of the labels. Shape: (batch_size, ).
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
    """Computes the kl divergence between policy and reference model.
    
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
    ignore_idx: Optional[int] = -100,
) -> Tuple[torch.FloatTensor, torch.LongTensor]:
    """Prepares the logits and labels for the given prompts and responses. 
    
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


class TYPOTrainer:
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
        """Intialize the TYPOTrainer.
        
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

        # optimizer 
        self.optimizer = AdamW(model.parameters(), lr=config.training.lr)
        
        # scheduler 
        self.scheduler = get_scheduler(
            name="linear", 
            optimizer=self.optimizer, 
            num_warmup_steps=config.training.num_warmup_steps, 
            num_training_steps=(
                len(self.train_dataloader) * self.config.training.n_epochs
                ) // config.training.gradient_accumulation_steps,
        )

        # writing checkpoints
        self.checkpoint_dir = config.training.checkpoint_dir
        print("Loaded model on rank", self.local_rank)
        print("Loaded reference model on rank", self.local_rank)
        print(f"Writing checkpoints to {self.config.training.checkpoint_dir}.")
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
        """Save model, optimizer, and scheduler state to disk, gathering from all processes and saving only on the rank 0 process. Copy-pasted from dpo repo."""
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
        reference_model: transformers.PreTrainedModel,
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Computes metrics for both policy and reference model."""
    
        # get logprobs for policy model 
        logits, labels = prepare_logits_labels(model, self.tokenizer, batch)
        
        batch_logprobs = _get_batch_logprobs(logits, labels)
        
        # reshape to be n_constitutions * n_responses
        batch_logprobs = batch_logprobs.view(self.config.data.n_constitutions,  self.config.data.n_responses)
        
        # get logprobs for reference model 
        with torch.no_grad():
            
            ref_logits, ref_labels = prepare_logits_labels(reference_model, self.tokenizer, batch)

        # typo loss
        loss = typo_loss(logprobs=batch_logprobs)
                                              
        return loss, batch_logprobs, logits, ref_logits
    
    def _run_batch(
        self, 
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[torch.FloatTensor, Dict[str, torch.Tensor], torch.FloatTensor]:
        """Run a batch."""    
        
        # compute policy and reference metrics
        loss, batch_logprobs, logits, ref_logits  = self.compute_metrics(self.model, self.reference_model, batch)
            
        kl_div = kl_divergence(
            policy_logits=logits,
            ref_logits=ref_logits,
        )
      
        # loss = loss + beta * kl
        adjusted_loss = loss + self.config.typo.beta * kl_div

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
                if torch.isnan(loss).any() or torch.isnan(raw_loss).any() or torch.isnan(kl_div).any():
                    if n_batches > 0:
                        average_loss = total_loss / n_batches
                        total_loss += average_loss
                        average_raw_loss = total_raw_loss / n_batches
                        total_raw_loss += average_raw_loss
                        average_kl_div = total_kl_div / n_batches
                        total_kl_div += average_kl_div
                    else:
                        total_loss += 0  
                        total_raw_loss += 0  
                        total_kl_div += 0 
                else:
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
                
                # hard-coded for 2 * 2 case atm. 
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
                
                # evaluate and save after n steps have been made
                if (step + 1) % self.config.training.save_after_n_steps == 0:
                    if self.config.training.evaluate:
                        self.evaluate()
                    self.save_checkpoint(round(step / len(self.train_dataloader), 2))
                                
        # evaluate at end of each epoch and save checkpoint 
        if self.config.training.evaluate:
            self.evaluate()
        self.save_checkpoint(epoch)