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
    "generation_prompt_1": """You are creating a list of human preferences for interacting with an AI assistant. This list should be based on analyzing 'PREFERRED' versus 'REJECTED' responses in conversations with an AI Assistant.

Current List of Preferences:

{constitution}

Analyze the new conversation pairs provided below. Each conversation includes a question asked by a human, followed by two different final responses from the AI assistant:

{conversations}

Guidelines for Analysis:
1. Focus on why the 'PREFERRED' response was more favorable compared to the 'REJECTED' one. Consider context, tone, and content.
2. Look for patterns or themes in the 'PREFERRED' responses that can translate into general preferences.

Revising the List:
1. Based on your analysis, revise the list of preferences. Add new preferences or adjust the existing ones to align with the insights from the 'PREFERRED' responses. 
2. If you come to the conclusion that the current list of preferences already captures the newly observed 'PREFERRED' responses, you do not need to change it.
3. Ensure all additions or changes to the list are directly drawn from specific details in the 'PREFERRED' responses, avoiding generic preferences not supported by observed evidence.
4. The list should be no longer than 10 principles overall.

Format for Revised List (do not refer to the conversations themselves in the list):

Revised List of Preferences:
1. """,
}