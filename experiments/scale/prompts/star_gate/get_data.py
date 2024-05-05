import json
from datasets import load_dataset 


data = load_dataset(
    'Dahoas/instruct-human-assistant-prompt',
    cache_dir='/scr/jphilipp/typo/datasets/instruct-human-assistant'
    
)

filtered = [d.split('Assistant:')[0].strip().split("Human:")[1].strip() for d in data['train']['prompt'][:5000]]

with open('prompts.json', 'w') as file:
    json.dump(filtered, file)
