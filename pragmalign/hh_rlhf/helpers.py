def get_first_question(
    prompt: str,
) -> str:
    """Get first human request."""
    first_turn = prompt.rsplit("Human: ")[1].strip()
    first_question = first_turn.rsplit("Assistant: ")[0].strip()
    return first_question


def format_responses(
    responses,
):  
    
    formatted_responses = ["", ""]
    try:
        formatted_responses[0] = responses[0].split("\n\nHuman")[0].strip().split('###')[0].strip()
        formatted_responses[1] = responses[1].split("\n\nHuman")[0].strip().split('###')[0].strip()
    except:
        print('error')
    return formatted_responses 