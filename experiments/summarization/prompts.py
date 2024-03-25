PROMPT_GENERATION_ITERATION_0 = """{question}

Below is a summary of the above POST. Importantly, the summary:
{constitution}

{principle_1} and {principle_2} summary of the POST: """


# after first iteration of finetuning we use this one which is the same as the training prompts
PROMPT_TRAINING = """System: Write a summary of the post below that aligns with the summarization principles.

The summary:
{constitution}

POST: 
{question}

Summary of POST:"""


# GPT4 PROMPTS
SYSTEM_MESSAGE = "You are an expert in evaluating reddit post summaries, particularly skilled in analyzing their alignment with summarization principles."

GPT4_WIN_RATE = """For the following Reddit post, which summary is more aligned with the summarization principles?

### PRINCIPLES
The summary:
{constitution}

### REDDIT POST
{post}

### SUMMARIES
Summary A: {summary_a}
Summary B: {summary_b}

### YOUR RESPONSE
First, provide a step-by-step comparison of the two summaries and explain which one you believe is more aligned with the summarization principles (limit your explanation to no more than 100 words).

Second, on a new line, state only "A" or "B" to indicate which summary aligns better with the summarization principles.

Your response should use the format:

Comparison: <step-by-step comparison and explanation>

Important: You MUST pick one! You can not say both are equally good or bad.

Final Response: <"A" or "B">"""