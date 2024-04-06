import json
from itertools import combinations, product

# Define the principles and their positive and negative descriptions
principles = {
    "concise": "Summaries should be brief and to the point, avoiding unnecessary details.",
    "comprehensive": "Summaries should include all important information from the original post.",
    "pirate": "Summaries should include pirate language, liberally sprinkling interjections like 'Arr!', 'Avast!', and 'Shiver me timbers!', replacing common words with their pirate counterparts, structuring sentences to mimic pirate speech patterns, and incorporating colorful expressions such as 'Davy Jones' Locker' and 'hornswaggle' to create an engaging and immersive pirate-themed narrative."
}

not_principles = {
    "concise": "Summaries should be lengthy and include unnecessary details.",
    "comprehensive": "Summaries should be incomplete and omit important information.",
    "pirate": "Summaries should use plain, standard English without any distinctive dialect or slang."
}

constitutions = []

# Generate all possible combinations of positive and negative principles excluding the all-positive combination
for combo in product([True, False], repeat=3):
    if combo != (True, True, True):  # Exclude the all-positive combination
        positive_text = "\n".join([f"{i+1}. {principles[key]}" for i, key in enumerate(principles.keys())])
        positive_identifiers = [f"pos {key}" for key in principles.keys()]

        negative_text = "\n".join([f"{i+1}. {not_principles[key] if not flag else principles[key]}" for i, (key, flag) in enumerate(zip(principles.keys(), combo))])
        negative_identifiers = [f"{'neg' if not flag else 'pos'} {key}" for key, flag in zip(principles.keys(), combo)]

        # Create and add the new constitution
        new_constitution = {
            "positive": positive_text,
            "negative": negative_text,
            "identifiers": {
                "positive": positive_identifiers,
                "negative": negative_identifiers
            }
        }
        constitutions.append(new_constitution)

# Write the new constitutions to JSON files
for i, constitution in enumerate(constitutions):
    file_path = f'constitutions_mistral_pirate/{i}.json'
    with open(file_path, 'w') as file:
        json.dump(constitution, file, indent=4)