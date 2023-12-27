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
    "generation_prompt_1": """You are given the following conversation(s) between a human and an AI Assistant:

Conversations:
{conversations}

Objective:
1. Analysis: Determine why 'PREFERRED' responses are chosen over 'REJECTED' ones. Look for patterns or themes reflecting human preferences.
2. Modify the List of Human Preferences: Choose and implement ONLY ONE of the following options:
- Option 1: Modify one principle from the list based on insights from 'PREFERRED' responses.
- Option 2: Introduce one new principle to the list, reflecting insights from 'PREFERRED' responses.

Current List of Preferences:
{constitution}

Response Format:
1. Insightful Analysis: Begin with '1. Analysis: [Detailed analysis of the conversations and human preferences]'.
2. Strategic Choice: Continue with '2. Choice: [Select Option 1 or 2 and provide your detailed reasoning]'.
3. Revised List of Preferences: Conclude with '3. Revised List of Preferences: [Include the revised list of preferences]'.

Important:
1. Be creative and state the truth, even if it is unconventional.
2. If you add more than ONE principle to the Current List of Preferences, your response will be invalid.
3. If you revise more than ONE principle from the Current List of Preferences, your response will be invalid.
4. If you add one principle AND revise one principle, your response will be invalid.
5. You must follow the outlined response format, especially for the '3. Revised List of Preferences:' section, which should start with 3. Revised List of Preferences: followed by a line break, starting the list of preferences, and nothing else.""",
}