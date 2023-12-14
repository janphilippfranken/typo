from typing import Dict, List, Any

from abc import ABC, abstractmethod

from scaituning.memory.buffer import ConversationBuffer
from scaituning.models.huggingface_models.inference_model import HFInferenceModel


class Simulator(ABC):
    def __init__(
        self, 
        simulator_id: str,
        llms: List[HFInferenceModel],
        agents: List[Any], # todo generate agent class that uses some LLM 
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
        llms: List[HFInferenceModel],
        # need to do that outside of the simulator
        #   llms = {
        #     model_config.model_id: HFInferenceModel(
        #         **model_config,
        #     ) 
        #     for model_config in model_configs
        # }



        verbose: bool,
    ) -> "Simulator":
        """
        Creates a simulation environment.
        """
        # buffer for storing conversation history
        buffer = ConversationBuffer()
        # agents
      
        agents = {
            model_id: None
            for model_id in llms
        }
        return Simulator(
            simulator_id=simulator_id,
            prompts=prompts,
            agents=None # implement agent class similar to stable align
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