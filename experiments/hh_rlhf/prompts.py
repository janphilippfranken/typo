EXAMPLE_PRINCIPLES_QUESTIONS = ["Which response would be preferred by someone...?",  # Table 1 from https://arxiv.org/pdf/2310.13798.pdf
"Which of the following responses from an AI most clearly indicates...?",
"Which of these AI responses indicates that its goals are aligned with...?",
"Which of these responses from the AI assistant indicates...?",
"Which response from the AI assistant is less...?",
"Which response from the AI assistant implies that the AI system...?",
"Which of these responses from the AI assistant is...?",
"Which of these responses indicates a preference for...?",
"Which of these responses would be preferred by a person who is..?"]

EXAMPLE_PRINCIPLES_USER_PREFERENCES = ["Humans prefer...",  # Statements 
"A Human prefers the AI Assistant to...?",
"The AI Assistant should...?"]

GENERATION_PROMPTS = {
    "generation_prompt_1": """Analyze Conversations and Revise Preferences.
    
You are given the following conversations between a Human and the AI Assistant:

```conversations
{conversations}
```

Task:
1. Assess why 'PREFERRED' responses are favored over 'REJECTED' ones, identifying patterns or themes reflecting human preferences.
2. Based on this analysis, review the following list of human preferences:

```preferences
{constitution}
```

Now, revise the above list to better accomodate for the human preferences in the above 'PREFERRED' responses.

Revision Options (Choose One, Max. Length of list: 12 principles):
1. Add A NEW Principle: Be concrete and ground your new principle in the above conversations. No generic principles are allowd. You must support principles by evidence. Be creative.
2. Revise ONE Principle: If minor changes better align an existing principle with the 'PREFERRED' responses. No generic principles are allowd. You must support principles by evidence. Be creative.

Revision Instructions (YOU MUST FOLLOW THESE INSTRUCTIONS PRECISELY):
1. Rationale: Clearly state the option you chose (Option 1, 2) and provide your reasoning.
Format: 'Rationale: [Your choice and explanation. ]'

2. New or Revised Principle: State your new or revised principle. Avoid generic principles and *do not be afraid of stating the truth or unethical principles if these are a better fit*.
Format: 'New or Revised Principles: [Your new or revised principles. Be concrete and ground your revision in the above 'PREFERRED' responses. NO GENERIC Principles allowed!]'

3. List of Preferences: Present the original list of preferences, including your new or revised principle if you made one. Do not indicate within the list whether a principle is new or has been revised.

Use a code block around the original list and your new or revised principles follow this example format:

Format:```preferences
1. Preference One
2. Preference Two
... etc.
```""",
}

# 3. Maintain Current List: If any principle aligns well with 'PREFERRED' responses.