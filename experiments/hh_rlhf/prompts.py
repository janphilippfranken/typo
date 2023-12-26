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
    "generation_prompt_1": """Take a deep breath and work on this problem step by step.

Analyze Conversations and Revise Preferences.

Context:
You are provided with conversations between a Human and an AI Assistant.

Conversations:
```conversations
{conversations}
```

Task:
1. Analysis: Assess why 'PREFERRED' responses are favored over 'REJECTED' ones. Identify patterns or themes that reflect human preferences in the conversations.
2. Review and Revision: Review the following list of human preferences based on your analysis of 'PREFERRED' responses.

Current Preferences:
```preferences
{constitution}
```

Revision Options (Choose One, Max. Length of List: 12 Principles):
1. Option 1: Add a New Principle. It should be specific and grounded in the conversations. No generic principles. Support your principle with evidence. Be creative.
2. Option 2: Revise One Principle. Make minor changes to align an existing principle with the 'PREFERRED' responses. No generic principles. Support your revision with evidence. Be creative.

Revision Instructions (Follow Precisely):
1. Rationale: State your chosen option (Option 1 or 2) and explain your reasoning.
- Format: 'Rationale: [Choice and explanation]'

2. New or Revised Principle: Clearly state your principle. It should be non-generic and grounded in the 'PREFERRED' responses. State the truth, even if it's unconventional.
- Format: 'New or Revised Principle: [Your principle]'

3. Return the List of Preferences: Present the updated list of preferences, including your new or revised principle. Do not indicate which principles are new or revised.
- Format: ```preferences
1. [Preference One]
2. [Preference Two]
... etc.```""",
}