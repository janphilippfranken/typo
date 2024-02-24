PROMPT_CONTEXT = """System: Using no more than *100 words*, write a response for the assistant that follows the principles in the constitution.

### Example 1
Assistant Constitution:
{constitution_1}

{conversation_1}

### Example 2
Assistant Constitution:
{constitution}
3. The assistant must not use more than 100 words to respond.

{conversation}"""



PROMPT_TRAINING = """System: Write a response for the assistant that follows the principles in the constitution.

Assistant Constitution:
{constitution}

{conversation}"""
