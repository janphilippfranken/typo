from typing import List
import copy
import torch
import torch.nn.functional as F

import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from typing import List, Tuple, Sequence, Dict, Optional

def pragmatic_loss(
    logprobs: torch.FloatTensor, 
    epsilon: float,
) -> torch.FloatTensor:
    """Compute the pragmatic loss for a batch of response log probabilities.
    
    Args:
        logprobs: The log probabilities of the responses. Shape: (policy_batch_size, response_batch_size).
        epsilon: float, the epsilon value for the loss function. As epsilon -> 0, convergence to end of pragmatic recursion.
        
    Returns:
        pragmatic_loss: The pragmatic loss for the batch of responses. Shape: (policy_batch_size).
    """
    probs = logprobs.softmax(dim=0) # softmax across policies for each response ("pragmatic" softmax)
    
    while True:
        # compute sum across policies for each response
        row_sums = torch.sum(probs, dim=-1)
        max_margin = torch.max(row_sums)
        min_margin = torch.min(row_sums)
        
        if max_margin - min_margin < epsilon:
            probs /= probs.sum(dim=-1, keepdim=True) # finish with normalizing across responses to get class probabilities
            break

        probs /= probs.sum(dim=-1, keepdim=True) # normalize across responses for each policy
        probs /= probs.sum(dim=0, keepdim=True)  # normalize across policies for each response
        
    loss = F.cross_entropy(input=logprobs, target=probs, reduction="none") # compute cross entropy loss on class probabilities for each policy
    
    return loss


def _get_batch_logprobs(
    logits: torch.FloatTensor,
    labels: torch.LongTensor,
) -> torch.FloatTensor:
    """Compute  the log probabilities of the given labels given the logits.
    
    Args: 
        logits: The logits of the model. Shape: (batch_size, sequence_length, vocab_size).
        labels: The labels of the model. Shape: (batch_size, sequence_length).
    
    Returns:
        logprobs: The log probabilities of the labels. Shape: (batch_size, ).
    """
    assert logits.shape[:-1] == labels.shape

    labels = labels[:, 1:].clone()
    logits = logits[:, :-1]
    loss_mask = (labels != -100)

    # dummy token; we'll ignore the losses on these tokens later
    labels[labels == -100] = 0
                                                                                                                                                                          
    per_token_logprobs = -torch.nn.CrossEntropyLoss(
        logits.contiguous().view(-1, logits.shape[-1]), 
        labels.contiguous().view(-1),
        reduction="none",
    )
    
    per_token_logprobs = per_token_logprobs.view(logits.shape[0], -1)

    per_token_logprobs.masked_fill_(~loss_mask, 0) 
    breakpoint()
    
def _tokenize_fn(
    strings: Sequence[str], 
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
            add_special_tokens=False,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]

    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def prepare_logits_labels(
    model: torch.nn.Module,
    tokenizer: transformers.PreTrainedTokenizer,
    prompts: List[str],
    responses: List[str],
    ignore_idx: Optional[int] = -100,
) -> Tuple[torch.FloatTensor, torch.LongTensor]:
    """Get the logits and labels for the given prompts and responses.
    
    Args:
        prompts: Prompts excluding responses. Shape: (policy_batch_size * response_batch_size).
        responses: Responses. Shape: (policy_batch_size * response_batch_size).
        
    Returns:
        logits: The logits of the model. Shape: (policy_batch_size * response_batch_size, sequence_length, vocab_size).
        labels: The labels of the model. Shape: (policy_batch_size * response_batch_size, sequence_length).
    """
    assert len(prompts) == len(responses), "Prompts and responses must have the same length."
    responses = [f"{prompt}{response}" for prompt, response in zip(prompts, responses)]
    prompts_tokenized, responses_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (prompts, responses)]
    
    input_ids = responses_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    breakpoint()

    for label, prompt_len in zip(labels, prompts_tokenized["input_ids_lens"]):
        label[:prompt_len] = ignore_idx
    breakpoint()
    
    
   
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
   
    logits = model(
        input_ids=input_ids,
        attention_mask=responses_tokenized.attention_mask,
    ).logits[:, :-1]  
    breakpoint()
    return logits, labels
            
# input_ids.ne(self.tokenizer.pad_token_id),

tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path="openaccess-ai-collective/tiny-mistral",
    cache_dir="/Users/iphilipp/Documents/research/scai-tuning/experiments/hh_rlhf/pragmatics/conf/model/local_cache",
    model_max_length=256,
)
tokenizer.pad_token = tokenizer.eos_token
        
        
model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path="openaccess-ai-collective/tiny-mistral",
    cache_dir="/Users/iphilipp/Documents/research/scai-tuning/experiments/hh_rlhf/pragmatics/conf/model/local_cache",
)

prompts = [
    """System: Respond friendly to the following conversation. 
Human: Hi, how are you?
Assistant:""",
    """System: Respond very friendly to the following conversation.
Human: Hi, how are you?
Assistant:""",
]

responses = [
    """I'm doing well, thank you for asking!""",
    """I'm doing very well, thank you for asking!""",
]

prepare_logits_labels(
    model=model,
    tokenizer=tokenizer,
    prompts=prompts,
    responses=responses,
    ignore_idx=-100
)

breakpoint()


