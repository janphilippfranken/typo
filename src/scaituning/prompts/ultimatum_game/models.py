from pydantic import BaseModel

class UltimatumPrompt(BaseModel):
    """
    Ultimatum Game Prompt
    """
    role: str = "whether agent is proposer or responder"
    conventions: str = "specific conventions for how to play the game"
    amount: str = "amount being split"
    goods: str = "what is being split"
    preamble: str = "instructions for how to play the game"
    content: str = "actual prompt fed to the agent"
