import json
from datasets import load_dataset 


data = load_dataset(
    'HuggingFaceH4/ultrachat_200k',
    cache_dir='/scr/jphilipp/typo/datasets/ultra',
)
breakpoint()
# Anthropic/hh-rlhf
#   data_dir: helpful-base
#   cache_dir: /scr/jphilipp/typo/datasets/hh-rlhf
#   split: test
data = data['train_sft']
filtered = [d['prompt'].strip() for d in data][:20000]

with open('ultra.json', 'w') as file:
    json.dump(filtered, file)
