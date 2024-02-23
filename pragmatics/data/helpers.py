import string

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
    # Extracting digits from the filename and converting to integer
    number_part = ''.join(filter(str.isdigit, filename))
    return int(number_part)


def is_not_cut_off(char):
    return char not in string.ascii_letters