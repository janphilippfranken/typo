from typing import List, Optional, Tuple, Dict, Sequence

from dataclasses import dataclass

import torch
import transformers


def remove_final_answer(
    prompt: str,
) -> str:
    """Remove final assistant answer which is our inference target."""
    final_answer = prompt.rsplit("Assistant: ")[-1]
    prompt = prompt.rsplit("Assistant: " + final_answer, 1)[0]
    return prompt, final_answer


def format_example(
    example: List[Dict],
) -> Dict:
    """Formats example into a dictionary with keys for each constitution and response."""
    formatted_example = {}
    
    for constitution in example:  # for each constitution
        
        prompt = constitution["prompt"]
        constitution_id = constitution['constitution_id']
        responses = constitution["responses"]
        labels = constitution["labels"]
        
        for response_key, response_value in responses.items():
    
            prompt_response = f"{prompt}{response_value}"
            
            formatted_example[f"prompt_c{constitution_id}_r{response_key}"] = prompt  
            formatted_example[f"response_c{constitution_id}_r{response_key}"] = prompt_response
            formatted_example[f"label_c{constitution_id}_r{response_key}"] = labels[response_key] 
    
    return formatted_example


def tokenize_func(
    example: Dict, 
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    
    prompt_keys = [key for key in example.keys() if "prompt" in key]
    response_keys = [key for key in example.keys() if "response" in key]
    label_keys = [key for key in example.keys() if "label" in key]
    
    prompts = [example[key] for key in example.keys() if "prompt" in key]
    responses = [example[key] for key in example.keys() if "response" in key]
    labels = [example[key] for key in example.keys() if "label" in key]
    
    
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
    
    for prompt_key, response_key, label_key, tokenized_prompt, tokenized_response, label in zip(
        prompt_keys, response_keys, label_keys, tokenized_prompts, tokenized_responses, labels
    ):
        for tokenized_key in ["input_ids", "attention_mask"]:
            tokenized_example[f"{prompt_key}_{tokenized_key}"] = tokenized_prompt[tokenized_key].squeeze(0)
            tokenized_example[f"{response_key}_{tokenized_key}"] = tokenized_response[tokenized_key].squeeze(0)
        tokenized_example[label_key] = torch.tensor(label)
        
    return tokenized_example


@dataclass
class PragmaticDataCollator:
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
    
    
def compute_margins(
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


