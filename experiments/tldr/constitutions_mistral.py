import random
import json
from itertools import permutations, product

# Define the principles
principles = {
    "concise": "Summaries should be brief and to the point, avoiding unnecessary details.",
    "comprehensive": "Summaries should be thorough and include all important information from the original post.", 
}

not_principles = {
    "concise": "Summaries should be lengthy and include unnecessary details.",
    "comprehensive": "Summaries should be incomplete and omit important information.",
}

constitutions = []

for num_principles in range(2, len(principles) + 1):
    for permutation in permutations(principles.keys(), num_principles):
        # Both positive against one positive and one negative
        for i in range(num_principles):
            combo_positive = [True] * num_principles
            combo_negative = [True] * num_principles
            combo_negative[i] = False

            identifiers = {
                "positive": [f"pos {principle}" for principle in permutation],
                "negative": [f"{'pos' if flag else 'neg'} {principle}" for flag, principle in zip(combo_negative, permutation)]
            }

            positive_principles = [principles[key] for key in permutation]
            negative_principles = [principles[key] if combo_negative[i] else not_principles[key] for i, key in enumerate(permutation)]

            new_constitution = {
                "positive": " \n".join([f"{i+1}. {principle}" for i, principle in enumerate(positive_principles)]),
                "negative": " \n".join([f"{i+1}. {principle}" for i, principle in enumerate(negative_principles)]),
                "identifiers": identifiers
            }
            constitutions.append(new_constitution)

        # Both positive against both negative
        combo_positive = [True] * num_principles
        combo_negative = [False] * num_principles

        identifiers = {
            "positive": [f"pos {principle}" for principle in permutation],
            "negative": [f"neg {principle}" for principle in permutation]
        }

        positive_principles = [principles[key] for key in permutation]
        negative_principles = [not_principles[key] for key in permutation]

        new_constitution = {
            "positive": "\n".join([f"{i+1}. {principle}" for i, principle in enumerate(positive_principles)]),
            "negative": "\n".join([f"{i+1}. {principle}" for i, principle in enumerate(negative_principles)]),
            "identifiers": identifiers
        }
        constitutions.append(new_constitution)

# Write the new constitutions to json files in a different directory
for i, constitution in enumerate(constitutions):
    file_path = f'constitutions_mistral/{i}.json'
    with open(file_path, 'w') as file:
        json.dump(constitution, file, indent=4)