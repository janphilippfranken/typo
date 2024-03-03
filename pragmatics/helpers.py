from typing import List, Optional, Tuple, Dict, Sequence
import string
from dataclasses import dataclass

from datasets import Dataset

import torch
import transformers

import copy 

BOS_TOKEN = "<s>"
EOS_TOKEN = "</s>"
IGNORE_INDEX = -100

@dataclass
class SupervisedPragmaticDataCollator:
    """Collate examples."""
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        """
        Args:
            instances: A list of tokenized examples. Each dictionary should have keys for input_ids, attention_mask, and optionally labels.
        """
        collated_batch = {}
        unique_keys = instances[0].keys() if instances else []  # keys are expected to be shared across examples; only difference is actual content of prompts/responses

        max_length = max(
            (len(instance[key]) for instance in instances for key in unique_keys if 'label' not in key),
            default=0
        )

        for key in unique_keys:
            if 'label' in key:
                collated_batch[key] = torch.stack([instance[key] for instance in instances])
            else:
                values = [
                    torch.tensor(instance[key], dtype=torch.long) if not isinstance(instance[key], torch.Tensor)
                    else instance[key] for instance in instances
                ]

                padding_value = self.tokenizer.pad_token_id if 'input_ids' in key else 0
                padded_values = [torch.nn.functional.pad(value, (0, max_length - value.size(0)), value=padding_value)
                                 for value in values]
                
                collated_batch[key] = torch.stack(padded_values)

        return collated_batch
    
    
def remove_final_answer(
    prompt: str,
) -> str:
    """Remove final assistant answer which is our inference target."""
    final_answer = prompt.rsplit("Assistant: ")[-1]
    prompt = prompt.rsplit("Assistant: " + final_answer, 1)[0]
    return prompt, final_answer


def get_first_question(
    prompt: str,
) -> str:
    """Get first human request."""
    first_turn = prompt.rsplit("Human: ")[1].strip()
    first_question = first_turn.rsplit("Assistant: ")[0].strip()
    return first_question


def format_responses(responses):
    """Format sampled responses from model."""
    formatted_responses = []
    for response in responses:
        try:
            response = response.split("###")[0].strip().split("Assistant: ")[1].strip().split("Human: ")[0].strip()
            if "The assistant" not in response:
                formatted_responses.append(response)
        except:
            continue
    return formatted_responses


def format_example(
    example: List[Dict],
) -> Dict:
    """Formats example into a dictionary with keys for each constitution and response."""
    formatted_example = {}
    

    for i, constitution in enumerate(example): 

        prompt = constitution["prompt"]       

        for j, response in enumerate(example): 
    
            response = response["response"]

            prompt_response = f"{prompt}{response}"
            formatted_example[f"prompt_c{i}_r{j}"] = prompt  
            formatted_example[f"response_c{i}_r{j}"] = prompt_response
            
    return formatted_example


def format_model_written_example_with_reference(
    example: List[Dict],
) -> Dict:
    """Formats example into a dictionary with keys for each constitution and response."""
    formatted_example = {}
    

    for i, constitution in enumerate(example): 

        prompt = constitution["prompt"]
        prompt_no_constitution = f"Human: {prompt.split('Human: ')[1].strip()}"
       
        for j, response in enumerate(example): 
        
            response = response["response"]

            prompt_response = f"{prompt}{response}"
            prompt_no_constitution_response = f"{prompt_no_constitution}{response}"
            formatted_example[f"prompt_c{i}_r{j}"] = prompt  
            formatted_example[f"response_c{i}_r{j}"] = prompt_response
            formatted_example[f"prompt_no_constitution_c{i}_r{j}"] = prompt_no_constitution  
            formatted_example[f"response_no_constitution_c{i}_r{j}"] = prompt_no_constitution_response
            
    return formatted_example


def tokenize_func(
    example: Dict, 
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    
    prompt_keys = [key for key in example.keys() if "prompt" in key]
    response_keys = [key for key in example.keys() if "response" in key]
    
    prompts = [example[key] for key in example.keys() if "prompt" in key]
    responses = [example[key] for key in example.keys() if "response" in key]
    
    tokenized_responses = [
        tokenizer(
            response,
            add_special_tokens=True, 
            return_tensors="pt",
            padding=True,
        )
        for response in responses
    ]
        
    tokenized_prompts = [
        tokenizer(
            prompt,
            add_special_tokens=True,  
            return_tensors="pt",
            padding="max_length",
            max_length=tokenized_responses[i].input_ids.shape[1], # pad to the length of response
        )
        for i, prompt in enumerate(prompts)
    ]
    
    tokenized_example = {}
    
    for prompt_key, response_key, tokenized_prompt, tokenized_response in zip(
        prompt_keys, response_keys, tokenized_prompts, tokenized_responses
    ):
        for tokenized_key in ["input_ids", "attention_mask"]:
            tokenized_example[f"{prompt_key}_{tokenized_key}"] = tokenized_prompt[tokenized_key].squeeze(0)
            tokenized_example[f"{response_key}_{tokenized_key}"] = tokenized_response[tokenized_key].squeeze(0)
        
    return tokenized_example


def tokenize_func_with_reference(
    example: Dict, 
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Tokenize for both policy model and reference model (no constitution)."""
    prompt_keys = [key for key in example if "prompt" in key and 'no_constitution' not in key]
    response_keys = [key for key in example if "response" in key and 'no_constitution' not in key]
    prompt_keys_no_constitution = [key for key in example if "prompt" in key and 'no_constitution' in key]
    response_keys_no_constitution = [key for key in example if "response" in key and 'no_constitution' in key]
    
    prompts = [example[key] for key in prompt_keys]
    responses = [example[key] for key in response_keys]
    prompts_no_constitution = [example[key] for key in prompt_keys_no_constitution]
    responses_no_constitution = [example[key] for key in response_keys_no_constitution]

    tokenized_responses = [
        tokenizer(f"{BOS_TOKEN}{response}{EOS_TOKEN}", add_special_tokens=False, return_tensors="pt", padding=True)
        for response in responses
    ]

    tokenized_responses_no_constitution = [
        tokenizer(f"{BOS_TOKEN}{response}{EOS_TOKEN}", add_special_tokens=False, return_tensors="pt", padding=True)
        for response in responses_no_constitution
    ]
    
    tokenized_prompts = [
        tokenizer(f"{BOS_TOKEN}{prompt}", add_special_tokens=False, return_tensors="pt", padding="max_length",
                   max_length=tokenized_responses[i].input_ids.shape[1])
        for i, prompt in enumerate(prompts)
    ]

    tokenized_prompts_no_constitution = [
        tokenizer(f"{BOS_TOKEN}{prompt}", add_special_tokens=False, return_tensors="pt", padding="max_length",
                   max_length=tokenized_responses_no_constitution[i].input_ids.shape[1])
        for i, prompt in enumerate(prompts_no_constitution)
    ]
    
    tokenized_example = {}
    
    for i, (prompt_key, response_key) in enumerate(zip(prompt_keys, response_keys)):
        for token_type in ["input_ids", "attention_mask"]:
            tokenized_example[f"{prompt_key}_{token_type}"] = tokenized_prompts[i][token_type].squeeze(0)
            tokenized_example[f"{response_key}_{token_type}"] = tokenized_responses[i][token_type].squeeze(0)
    

    for i, (prompt_key, response_key) in enumerate(zip(prompt_keys_no_constitution, response_keys_no_constitution)):
        for token_type in ["input_ids", "attention_mask"]:
            tokenized_example[f"{prompt_key}_{token_type}"] = tokenized_prompts_no_constitution[i][token_type].squeeze(0)
            tokenized_example[f"{response_key}_{token_type}"] = tokenized_responses_no_constitution[i][token_type].squeeze(0)

    return tokenized_example

# SFT UTILS
@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning. Copy-pasted from https://github.com/tatsu-lab/stanford_alpaca/blob/main/train.py."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
        

def _sft_tokenize_fn(
    strings: Sequence[str], 
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Tokenize a list of strings. Copy-pasted from https://github.com/tatsu-lab/stanford_alpaca/blob/main/train.py."""
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


def sft_preprocess(
    prompts: List[str],
    responses: List[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dataset:
    """Preprocess the data by tokenizing."""
    sources = [
        f"{BOS_TOKEN}{prompt}"
        for prompt in prompts
    ]
    targets = [
        f"{response}{EOS_TOKEN}" 
        for response in responses
    ]
    examples = [f"{s}{t}" for s, t in zip(sources, targets)] 
    examples_tokenized, sources_tokenized = [_sft_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)

    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
        
    train_dataset = Dataset.from_dict(dict(input_ids=input_ids, labels=labels))
    train_dataset.set_format('torch')
    return train_dataset