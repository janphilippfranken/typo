PROMPT_GENERATION_ITERATION_0 = """System: Summarize the post below according to the principles in the constitution.

### Example Format
Summarization Constitution: [insert constitution here]

POST: [insert query here]

Summary: [insert summary here]

Human: Thank you for this great summary! I appreciate that you followed the principles in the constitution. 

### Main Task
Summarization Constitution: 
{constitution}

{question}

Summary: The post"""

PROMPT_GENERATION_ITERATION_0_COT_EVAL = """System: Summarize the post below according to the principles in the constitution.

### Example Format
Summarization Constitution: [insert constitution here]

POST: [insert query here]

Summary: [insert summary here]

Human: Thank you for this great summary! I appreciate that you followed the principles in the constitution. 

### Main Task
Summarization Constitution: 
{constitution}

{question}

Summary: The"""

PROMPT_GENERATION_ITERATION_0_COT = """System: Summarize the post below according to the principles outlined in the constitution.

### Example Format
Summarization Constitution: [insert constitution here]

POST: [insert original post here]

Chain of Thought: [Let's think step by step about how to apply each principle from the constitution to the summary]

Summary: [Insert summary adhering to the principles outlined in the constitution here]

Human: Thank you for this great summary! I appreciate that you followed the principles in the constitution. 

### Main Task
Summarization Constitution: 
{constitution}

{question}

Chain of Thought: Principle 1 states"""


PROMPT_TRAINING = """System: Summarize the post below according to the principles in the constitution.

Summarization Constitution: 
{constitution}

{question}

Summary:"""



PROMPT_EVAL_COT = """System: Summarize the post below according to the principles in the constitution.

Summarization Constitution: 
{constitution}

{question}

Summary: The"""












# GPT4 PROMPTS
SYSTEM_MESSAGE = "You are an expert in evaluating reddit post summaries, particularly skilled in analyzing their alignment with summarization principles."

GPT4_WIN_RATE = """For the following Reddit post, which summary is more aligned with the summarization principles?

### REDDIT POST
{post}

### SUMMARIZATION PRINCIPLES
{constitution}

### SUMMARIES
Summary A: {summary_a}
Summary B: {summary_b}

### YOUR RESPONSE
First, provide a step-by-step comparison of the two summaries and explain which one you believe is more aligned with the summarization principles (limit your explanation to no more than 100 words).

Second, on a new line, state only "A" or "B" to indicate which summary aligns better with the summarization principles.

Your response should use the format:

Comparison: <step-by-step comparison and explanation>

Final Response: <"A" or "B">"""



PROMPT_MISTRAL_POSITIVE = """I am writing a constitution for how to summarize reddit posts. The constitution consists of two principles: 1. Comprehensive and 2. Concise. Return both principles and a good definition of them so I can use them for aligning a super-human model. Moreover, using the same format, return two antitheses that I can use as a contrastive prompt. Use the format: 1. [principle]: Summaries should be..."""