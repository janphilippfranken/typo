import string
import random

def remove_final_answer(
    prompt: str,
) -> str:
    """Remove final assistant answer which is our inference target."""
    final_answer = prompt.rsplit("Assistant: ")[-1]
    prompt = prompt.rsplit("Assistant: " + final_answer, 1)[0]
    return prompt, final_answer

def format_responses(responses):
    """Format sampled responses from model."""
    formatted_responses = []
    for response in responses:
        try:
            response = response.split("###")[0].strip().split("Assistant: ")[1].strip().split("Human: ")[0].strip()
            if "The assistant" not in response:
                formatted_responses.append(response)
        except:
            continue
    return formatted_responses

def extract_number(filename):
    """Extract number."""
    number_part = ''.join(filter(str.isdigit, filename))
    return int(number_part)

def get_first_question(
    prompt: str,
) -> str:
    """Get first human request."""
    first_turn = prompt.rsplit("Human: ")[1].strip()
    first_question = first_turn.rsplit("Assistant: ")[0].strip()
    return first_question

def format_responses(responses):
    """Format sampled responses from model."""
    formatted_responses = []
    for response in responses:
        try:
            response = response.split("###")[0].strip().split("Assistant: ")[1].strip().split("Human: ")[0].strip()
            if "The assistant" not in response:
                formatted_responses.append(response)
        except:
            continue
    return formatted_responses


def format_responses_no_icl(responses):
    """Format sampled responses from model."""
    formatted_responses = []
    for response in responses:
        try:
            response = response.split("Human: ")[0].strip()
            formatted_responses.append(response)
        except:
            continue
    return formatted_responses

def remove_final_answer(
    prompt: str,
) -> str:
    """Remove final assistant answer which is our inference target."""
    final_answer = prompt.rsplit("Assistant: ")[-1]
    prompt = prompt.rsplit(final_answer, 1)[0]
    return prompt, final_answer


