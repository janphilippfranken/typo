import random
import json
from itertools import permutations, product



# Define the principles
principles = {
    "comprehensive": "Ensure summaries encompass all critical aspects, providing a full understanding without the need for further clarification.",
    "concise": "Keep it short and sweet, distilling information to its essence without unnecessary fluff.",
    "funny": "Infuse humor to the max, ensuring summaries not only enlighten but also entertain, possibly making the reader snort with laughter, while balancing clarity and insight."
}

# Define the NOT versions of the principles
not_principles = {
    "comprehensive": "Skim the surface, leaving out critical aspects and forcing the reader to seek further clarification elsewhere.",
    "concise": "Wander through the information wilderness, adding as much fluff as possible, making the summary long-winded and bloated.",
    "funny": "Keep humor at bay, ensuring summaries are dry and dull, potentially inducing yawns rather than laughter, while risking clarity for the sake of being overly serious."
}


constitutions = []

for permutation in permutations(principles.keys(), 3):
    contrasted_principle = permutation[-1]  # The last principle is always contrasted
    
    for combo in product([True, False], repeat=2):
        # Generating positive/negative text for the first two principles based on the combo flags
        pos_or_neg_text = ["pos" if flag else "neg" for flag in combo]
        # Creating identifiers for all principles, including the name and whether it's positive or negative
        identifiers = {
            "positive": [f"{pos_or_neg_text[i]} {permutation[i]}" for i in range(2)] + [f"pos {contrasted_principle}"],
            "negative": [f"{pos_or_neg_text[i]} {permutation[i]}" for i in range(2)] + [f"neg {contrasted_principle}"]
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

