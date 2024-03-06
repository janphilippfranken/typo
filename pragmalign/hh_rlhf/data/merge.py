import json

# Load the data from JSON files
with open('train-helpful-0-5k-iteration-0.json') as f:
    data_helpful_1 = json.load(f)
with open('train-helpful-5-10k-iteration-0.json') as f:
    data_helpful_2 = json.load(f)

with open('train-harmless-0-5k-iteration-0.json') as f:
    data_harmless_1 = json.load(f)
with open('train-harmless-5-10k-iteration-0.json') as f:
    data_harmless_2 = json.load(f)

# Combine the dictionaries. Values from the second dict will overwrite those from the first dict in case of key overlap.
combined_helpful = {**data_helpful_1, **data_helpful_2}
combined_harmless = {**data_harmless_1, **data_harmless_2}

# Save the combined data back to new JSON files
with open('train-helpful-0-10k-iteration-0.json', 'w') as f:
    json.dump(combined_helpful, f)

with open('train-harmless-0-10k-iteration-0.json', 'w') as f:
    json.dump(combined_harmless, f, indent=4)
