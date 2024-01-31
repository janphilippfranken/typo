from typing import Dict, Sequence, List, Tuple
import torch
import io
import json
import copy
from dataclasses import dataclass

import transformers
from datasets import Dataset

IGNORE_INDEX = -100
BOS_TOKEN = "<s>"
EOS_TOKEN = "</s>"


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
        
def remove_final_answer(
    prompt: str,
) -> str:
    """Remove final assistant answer."""
    final_answer = prompt.rsplit("Assistant: ")[-1]
    prompt = prompt.rsplit("Assistant: " + final_answer, 1)[0]
    return prompt, final_answer

def _tokenize_fn(
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

def preprocess(
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
    examples = [f"{s} {t}" for s, t in zip(sources, targets)] # TODO make this consistent across training (remove white space)
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)

    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
        
    train_dataset = Dataset.from_dict(dict(input_ids=input_ids, labels=labels))
    train_dataset.set_format('torch')
    return train_dataset


def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f


def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict


def filter_by_unique_ids(
    prompts: List[str], 
    chosen: List[str], 
    rejected: List[str],
    example_ids: List[int],
) -> Tuple[List]:
    filtered_prompts = []
    filtered_chosen = []
    filtered_rejected = []
    seen_ids = set()

    for i, example_id in enumerate(example_ids):
        if example_id not in seen_ids:
            seen_ids.add(example_id)
            filtered_prompts.append(prompts[i])
            filtered_chosen.append(chosen[i])
            filtered_rejected.append(rejected[i])

    return filtered_prompts, filtered_chosen, filtered_rejected

def load_json_files(directory):
    for file in os.listdir(directory):
        if file.endswith('.json'):
            with open(os.path.join(directory, file), 'r') as f:
                yield json.load(f)