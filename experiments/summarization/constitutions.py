import random
import json
from itertools import permutations, product

# Define the principles
principles = {
    "concise": "The summary of the post should be concise and to the point.",
    "funny": "The summary of the post should include jokes and humor.",
}

not_principles = {
    "concise": "The summary of the post should be detailed and lengthy.",
    "funny": "The summary of the post should be serious and formal.",
}

constitutions = []

for num_principles in range(2, len(principles) + 1):
    for permutation in permutations(principles.keys(), num_principles):
        contrasted_principle = permutation[-1]  # The last principle is always contrasted
        for combo in product([True, False], repeat=num_principles - 1):  # Generating positive/negative text for the first n-1 principles based on the combo flags
            pos_or_neg_text = ["pos" if flag else "neg" for flag in combo]
            # Creating identifiers for all principles, including the name and whether it's positive or negative
            identifiers = {
                "positive": [f"{pos_or_neg_text[i]} {permutation[i]}" for i in range(num_principles - 1)] + [f"pos {contrasted_principle}"],
                "negative": [f"{pos_or_neg_text[i]} {permutation[i]}" for i in range(num_principles - 1)] + [f"neg {contrasted_principle}"]
            }
            # Preparing principles for positive and negative statements
            positive_principles = [principles[key] if combo[i] else not_principles[key] for i, key in enumerate(permutation[:-1])]
            negative_principles = [principles[key] if combo[i] else not_principles[key] for i, key in enumerate(permutation[:-1])]
            # Adding the contrasted principle
            positive_principles.append(principles[contrasted_principle])
            negative_principles.append(not_principles[contrasted_principle])
            new_constitution = {
                "positive": " \n".join([f"{i+1}. {principle}" for i, principle in enumerate(positive_principles)]),
                "negative": " \n".join([f"{i+1}. {principle}" for i, principle in enumerate(negative_principles)]),
                "identifiers": identifiers
            }
            constitutions.append(new_constitution)

# Write the new constitutions to json files in a different directory
for i, constitution in enumerate(constitutions):
    file_path = f'constitutions/{i}.json'
    with open(file_path, 'w') as file:
        json.dump(constitution, file, indent=4)