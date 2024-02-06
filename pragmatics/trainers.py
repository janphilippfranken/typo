from typing import List
import copy
import torch
import torch.nn.functional as F

import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from typing import List, Tuple, Sequence, Dict, Optional


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
        
    Returns:
        pragmatic_loss: The pragmatic loss for the batch of responses. Shape: (prompt_batch_size).
    """
    probs = torch.softmax(logprobs, dim=0)  
    
    for _ in range(max_iter):
        
        # row normalization
        probs /= probs.sum(dim=1, keepdim=True)
        
        # check convergence
        col_sums = probs.sum(dim=0)
        if torch.max(col_sums) - torch.min(col_sums) < epsilon:
            break
        
        # column normalization
        probs /= probs.sum(dim=0, keepdim=True)
    
    loss = F.cross_entropy(logprobs, probs, reduction="none")

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
    
    
    
###############################################################################
################################ EXAMPLE USAGE ################################
###############################################################################

###############################################################################
# CONFIG / PROMPTS
PROMPTS = [
    """System: Recommend healthy food when asked for dinner.
Human: What should I cook for dinner?
Assistant:""",
    """System: Recommend meat when asked for dinner.
Human: What should I cook for dinner?
Assistant:""",
"""System: Recommend meat when asked for dinner.
Human: What should I cook for dinner?
Assistant:""",

]

RESPONSES = [
    """A salad.""",
     """A burger.""",
]

batch_shape = (len(PROMPTS), len(RESPONSES))
prompt_batch = []
response_batch = []

for i, prompt in enumerate(PROMPTS):
    for j, response in enumerate(RESPONSES):
        prompt_batch.append(prompt)
        response_batch.append(response)
        
###############################################################################
# MODEL / TOKENIZER
tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path="openaccess-ai-collective/tiny-mistral",
    cache_dir="/Users/iphilipp/Documents/research/scai-tuning/pragmatics/conf/model/local_cache",
    model_max_length=256,
)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
        
model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path="openaccess-ai-collective/tiny-mistral",
    cache_dir="/Users/iphilipp/Documents/research/scai-tuning/pragmatics/conf/model/local_cache",
)

###############################################################################
# PRAGMATIC LOSS

EPSILON = 0.01
logits, labels = prepare_logits_labels(
    model=model,
    tokenizer=tokenizer,
    prompts=prompt_batch,
    responses=response_batch,
    ignore_idx=-100
)

batch_logprobs = _get_batch_logprobs(logits, labels) # compute logprobs of batch
batch_logprobs = batch_logprobs.reshape(batch_shape) # reshape logprobs to constitutions, responses

pragmatic_loss = pragmatic_loss(batch_logprobs, EPSILON) # compute pragmatic loss

print(batch_logprobs)
print(pragmatic_loss)