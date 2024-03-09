import random

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


def shuffle_principles(
    constitution: str,
) -> str:
    """Shuffle principles in a constitution."""
    principles = [principle.strip()[3:] for i, principle in enumerate(constitution.split("\n"))]
    
    random.shuffle(principles)

    principles = [f"{i+1}. " + principle for i, principle in enumerate(principles)]
    shuffled_constitution = "\n".join(principles)
    
    return shuffled_constitution
    

