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
    "generation_prompt_1": """You are tasked with revising a list of human preferences for interacting with an AI assistant. Your revisions should stem from an in-depth analysis of 'PREFERRED' and 'REJECTED' responses in conversations with an AI Assistant.

Current List of Preferences:

{constitution}

New Conversation Pairs for Analysis:

{conversations}

Guidelines for Analysis:
1. For each response, evaluate why the 'PREFERRED' response was more favorable than the 'REJECTED' one.
2. Identify any patterns or themes in the 'PREFERRED' responses that reflect underlying human preferences.

Revision Instructions (Pick One Only):
1. Revising an Existing Principle: If an existing principle can be slightly modified to better reflect new insights from the 'PREFERRED' responses, make a minor revision to that principle. Focus on minor adjustments only and avoid a complete rephrasing.
2. Add a New Principle: If none of the existing preferences capture the new insights gained from the 'PREFERRED' responses, add one new principle to the list.
3. Maintain the Current List (Default Response): If you determine that one or multiple principles in the current list already adequately align with the 'PREFERRED' responses, do not change the current list. This should be your default response unless you are confident that amending or adding a new principle is the better choice. Even if only one existing principle aligns well, no revisions or additions are necessary.

IMPORTANT: 
1. Be very conservative with your revisions. Only make changes if they significantly enhance the list's alignment with the 'PREFERRED' responses.
2. Limit your action to revising or adding only ONE PRINCIPLE. You cannot modify or add more than one principle at a time, even if the current list is empty.
3. Your decision should be based on a solid rationale drawn from your analysis and must not be generic.

Return Format:
Provide the Current List of Preferences with your revisions, formatted as follows:

Revised List of Preferences:
1. ...

Include only the current list of preferences (if any) in the revised list as well as your edits or new preference (if any).
Do not provide commentary on the changes made.""",
}