from typing import Dict, List, Any

from abc import ABC, abstractmethod

from scaituning.memory.buffer import ConversationBuffer
from scaituning.models.hf_inference_model import HFInferenceModel


class Simulator(ABC):
    def __init__(
        self, 
        simulator_id: str,
        agents: List[HFInferenceModel],
        prompts: Dict[Any], 
        buffer: ConversationBuffer, 
        verbose: bool, 
    ) -> None:
        """
        Initializes a simulator. 

        Args:
            simulator_id: ID of the simulator.
            agents: List of agents.
            prompts: Dictionary of prompts for different roles/agents.
            buffer: Conversation buffer for storing conversations.
            verbose: Whether to print out information during simulation.
        """
        self.simulator_id = simulator_id
        self.agents = agents
        self.prompts = prompts
        self.buffer = buffer
        self.verbose = verbose

    @staticmethod
    def create(
        simulator_id: str,
        model_configs: List[Dict],
        prompts: Dict[str],
        verbose: bool,
    ) -> "Simulator":
        """
        Creates a simulation environment.
        """
        # buffer for storing conversation history
        buffer = ConversationBuffer()
        # agents
        agents = {
            model_config.model_id: HFInferenceModel(
                **model_config,
            ) 
            for model_config in model_configs
        }
        return Simulator(
            simulator_id=simulator_id,
            prompts=prompts,
            agents=agents,
            buffer=buffer,
            verbose=verbose,
        )

    @abstractmethod
    def step(
        self,
        step: int,
    ) -> None:
        """
        Runs a step of the simulation.

        Args:
            step: Step number.
        """
        raise NotImplementedError

    @abstractmethod
    def run(
        self, 
        n_steps: int,
    ) -> None:
        """
        Runs a simulation for n_steps.

        Args:
            n_steps: Number of steps to run the simulation for.
        """
        raise NotImplementedError