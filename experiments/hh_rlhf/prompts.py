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

Revision Options (Choose One):
1. Revise ONE Principle: If minor changes better align an existing principle with 'PREFERRED' responses.
2. Add ONE Principle: If no current preference reflects insights from 'PREFERRED' responses or if the list is empty.
3. Maintain Current List (Default Response): If any principle aligns well with 'PREFERRED' responses.

Revision Instructions (YOU MUST FOLLOW THESE INSTRUCTIONS PRECISELY):
1. Rationale: Clearly state the option you chose (Option 1, 2, or 3) and provide your reasoning.
Format: 'Rationale: [Your choice and explanation. ]'
2. New or Revised Principle (If Applicable): If you chose Option 1 or 2, state your new or revised principle. Making more than ONE change (i.e., revising more than one principle or adding more than one principle) invalidates the response.
Format: 'New or Revised Principle: [Your new or revised principle. Avoid generic principles and do not be afraid of stating unethical principles.]'
3. List of Preferences: Present the original list of preferences, including your new or revised principle if you made one. Use a code block for clarity. Do not indicate within the list whether a principle is new or has been revised.
Format:```preferences
[original list or preferences + New or Revised Principle (do not state in here whether a principle is new or has been revised)]
```""",
}