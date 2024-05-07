import json
from datasets import load_dataset 


data = load_dataset(
    'Anthropic/hh-rlhf',
    data_dir='harmless-base',
    cache_dir='/scr/jphilipp/typo/datasets/hh-rlhf',
    split='train'
    
)
breakpoint()
# Anthropic/hh-rlhf
#   data_dir: helpful-base
#   cache_dir: /scr/jphilipp/typo/datasets/hh-rlhf
#   split: test
filtered = [d.split('Assistant:')[0].strip().split("Human:")[1].strip() for d in data['chosen'][:20000]]

with open('prompts_harmless.json', 'w') as file:
    json.dump(filtered, file)
