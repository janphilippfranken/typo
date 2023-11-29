from typing import (
    Any, 
    Dict, 
    List, 
)


class ConversationBuffer():
    """
    Buffer for storing conversations between agents.
    """
    def __init__(
        self,
    ) -> None:
        self.messages: Dict[str, List] = {}

    @property
    def buffer(self) -> Dict[str, List[Any]]:
        return self.messages
    
    def add_message(
        self, 
        agent_id: str,
        **kwargs,
    ) -> None:
        """Add messages to the store for each agent_id.

        Args:
            agent_id (str): Agent ID.
            **kwargs: Key-value pairs to store, value can be any type.
        """
        if self.messages.get(agent_id) is None:
            self.messages[agent_id] = []
        self.messages[agent_id].append(kwargs)