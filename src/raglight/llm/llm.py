from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict


class LLM(ABC):
    """
    Abstract base class for large language models (LLMs).

    This class serves as a blueprint for implementing various LLMs with a consistent interface
    for loading and generating text.

    Attributes:
        model_name (str): The name of the LLM model.
        model (Any): The loaded model instance.
    """

    def __init__(self, model_name: str) -> None:
        """
        Initializes an LLM instance.

        Args:
            model_name (str): The name of the LLM model to be loaded.
        """
        self.model_name: str = model_name
        self.model: Any = self.load()
        self.system_prompt: str = ""

    @abstractmethod
    def load(self) -> Any:
        """
        Abstract method to load the LLM model.

        This method must be implemented by subclasses to define the loading logic for a specific LLM.

        Returns:
            Any: The loaded model instance.
        """
        pass

    @abstractmethod
    def generate(self, input: Dict[str, Any]) -> str:
        """
        Abstract method to generate text using the LLM model.

        This method must be implemented by subclasses to define how the model generates output
        based on the given input.

        Args:
            input (Dict[str, Any]): A dictionary containing the input data for text generation. The structure
                                    and required keys depend on the specific LLM implementation.

        Returns:
            Any: The generated output from the model.
        """
        pass
