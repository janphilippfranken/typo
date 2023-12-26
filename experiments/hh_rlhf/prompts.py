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
    "generation_prompt_1": """Help me analyze conversations and revise a list of human preferences.

Context:
You are provided with conversations between a Human and an AI Assistant.

Conversations:
###############################
{conversations}
###############################

Task:
1. Analysis: Assess why 'PREFERRED' responses are favored over 'REJECTED' ones. Identify patterns or themes that reflect human preferences in the conversations.
2. Review and Revision: Review the following List of human preferences based on your analysis of 'PREFERRED' responses.

Old Preferences:
<OLD PREFERENCES START> 
{constitution}
<OLD PREFERENCES END>

Revision Options (Choose One, Max. Length of List: 9 Principles):
1. Option 1: Add ONE New Principle to the above list. It should be specific and grounded in the conversations.
2. Option 2 (only chose this one if old list has already 9 or more preferences): Revise One Principle from the above list. Make minor changes to better align one existing principle with the 'PREFERRED' responses.


Format your response as follows:

First, explain your reasoning and choice in:
'1. Let's Think Step By Step: [your reasoning]'

Next, state the new or revised principle in:
'2. Choice: Option [your choice and justification]'. No generic principles allowed. Only ADD ONE or REVISE ONE Principle at a time. Be creative and state the truth, even if it's unconventional.

Finally, return the old list of preferences including your new or revised principle formatted as a block (following the exact same format as the list above):
'3. <REVISED PREFERENCES START>
[OLD PREFERENCES + YOUR NEW OR REVISED PRINCIPLE]
<REVISED PREFERENCES END>'

Important:
1. Do not indicate which principles are new or revised.
2. If you do not follow the exact formatting above using the '<REVISED PREFERENCES START> <REVISED PREFERENCES END>' block to return the updated list (including the old preferences that were not revised), your response will be invalid.""",
}