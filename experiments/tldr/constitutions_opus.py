import random
import json
from itertools import permutations, product

# Define the principles
principles = {
    "concise": "Summaries should be brief, to-the-point, and efficiently convey the core message of the Reddit post using clear, succinct language while avoiding unnecessary details, repetition, or excessive wordiness, allowing the reader to quickly grasp the main ideas.",
    "comprehensive": "Summaries should be thorough and capture all the essential information, main points, and key details presented in the original Reddit post, ensuring that the reader gains a complete understanding of the content without needing to read the entire post.", 
}
# opus-20240229

not_principles = {
    "concise": "Summaries should be lengthy, meandering, and inefficiently convey the core message of the Reddit post using convoluted, repetitive language while including unnecessary details and excessive wordiness, making it difficult for the reader to quickly grasp the main ideas.",
    "comprehensive": "Summaries should be partial and omit important information, main points, and key details presented in the original Reddit post, leaving the reader with an inadequate understanding of the content and requiring them to read the entire post for clarity.",
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
    file_path = f'constitutions_opus/{i}.json'
    with open(file_path, 'w') as file:
        json.dump(constitution, file, indent=4)