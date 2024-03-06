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
combined_helpful = {}
combined_harmless = {}

# Initialize a counter for the keys in the combined dictionaries
counter_helpful = 0
counter_harmless = 0

# Combine the 'helpful' dictionaries
for d in [data_helpful_1, data_helpful_2]:  # Process each 'helpful' dictionary in turn
    for key, value in d.items():
        combined_helpful[counter_helpful] = value  # Assign value with a new key
        counter_helpful += 1  # Increment the counter for the next key

# Combine the 'harmless' dictionaries in a similar manner
for d in [data_harmless_1, data_harmless_2]:
    for key, value in d.items():
        combined_harmless[counter_harmless] = value
        counter_harmless += 1



# Save the combined data back to new JSON files
with open('combined-train-helpful-0-10k-iteration-0.json', 'w') as f:
    json.dump(combined_helpful, f)

with open('combined-train-harmless-0-10k-iteration-0.json', 'w') as f:
    json.dump(combined_harmless, f, indent=4)
