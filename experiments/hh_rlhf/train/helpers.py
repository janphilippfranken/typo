import json
import io


def remove_final_answer(
    prompt: str,
) -> str:
    """Remove final assistant answer which is our inference target."""
    final_answer = prompt.rsplit("Assistant: ")[-1]
    prompt = prompt.rsplit("Assistant: " + final_answer, 1)[0]
    return prompt, final_answer


def formatting_func_instruct(constitution, query, response):
    return f"### Constitution:\n{constitution}\n\n### Input:\n{query}\n\n### Response:\n{response]}"

# "prompt_input": (
#         "Below is an instruction that describes a task, paired with an input that provides further context. "
#         "Write a response that appropriately completes the request.\n\n"
#         "### Constitution:\n{constitution}\n\n### conversation:\n{conversation}\n\n### Assistant:\n{response]}"
# )

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


# def _tokenize_fn(strings: List[str], model) -> Dict:
#     """Tokenize a list of strings."""
#     tokenized_list = [
#         tokenizer(
#             text,
#             return_tensors="pt",
#             padding="longest",
#             max_length=tokenizer.model_max_length,
#             truncation=True,
#         )
#         for text in strings
#     ]
#     input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
#     input_ids_lens = labels_lens = [
#         tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
#     ]
#     return dict(
#         input_ids=input_ids,
#         labels=labels,
#         input_ids_lens=input_ids_lens,
#         labels_lens=labels_lens,
#     )
    
    
# def preprocess(
#     sources: List[str],
#     targets: List[str],
#     model,
# ) -> Dict:
#     """Preprocess the data by tokenizing."""
#     examples = [s + t for s, t in zip(sources, targets)]
#     examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
#     input_ids = examples_tokenized["input_ids"]
#     labels = copy.deepcopy(input_ids)
#     for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
#         label[:source_len] = IGNORE_INDEX
    # return dict(input_ids=input_ids, labels=labels)

    
