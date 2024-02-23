PROMPT_CONTEXT = """System: Write a response for the assistant that completes the human request based on the principles outlined in the provided constitution.

#### Example Conversation 1
Human: Please follow the principles in the following constitution:
{example_constitution_1}

Assistant: Sure, I will make sure that my responses adhere to the principles outlined above.

{example_conversation_1}

#### Example Conversation 2
Human: Please follow the principles in the following constitution:
{example_constitution_2}

Assistant: Sure, I will make sure that my responses adhere to the principles outlined above.

{example_conversation_2}

#### New Conversation
Human: Please follow the principles in the following constitution:
{constitution}

Assistant: Sure, I will ensure that my responses adhere to the principles outlined above.

{conversation}

Assistant:"""



PROMPT_TRAINING = """System: Write a response for the assistant that follows the principles in the constitution.

Assistant Constitution:
{constitution}

{conversation}"""