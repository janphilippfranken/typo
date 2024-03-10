from typing import List, Optional, Tuple, Dict, Sequence

from datasets import Dataset
from dataclasses import dataclass

import torch
import transformers

import copy 

EOS_TOKEN = "</s>"
BOS_TOKEN = "<s>"
IGNORE_INDEX = -100

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

        prompt = f"{constitution['prompt']}\n\nAssistant:"
        # # \n\nAssistant:"
        # print("CHECK ASSISTANT PROMPT")
        # breakpoint()

        for j, response in enumerate(example): 
    
            response = response["response"]

            prompt_response = f"{prompt}{response}"
            formatted_example[f"prompt_c{i}_r{j}"] = prompt  
            formatted_example[f"response_c{i}_r{j}"] = prompt_response
            
    return formatted_example


def format_example_dpo(
    example: List[Dict],
) -> Dict:
    """Formats example for dpo data."""
    formatted_example = {'prompts': [], 'chosen': [], 'rejected': []}
    

    for i, constitution in enumerate(example): 

        prompt = f"{constitution['prompt']}\n\nAssistant:"
        # print("CHECK ASSISTANT PROMPT")
        # breakpoint()
        
        formatted_example['prompts'].append(prompt)    

        for j, response in enumerate(example): 
    
            response = response["response"]
        
            if i == j:
                formatted_example['chosen'].append(response)
            
            elif i != j:
                formatted_example['rejected'].append(response)
            
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