from __future__ import annotations
from typing import Optional, Dict, Any
from typing_extensions import override
from ..config.settings import Settings
from .llm import LLM
from mistralai import Mistral
from json import dumps
import logging


class MistralModel(LLM):
    """
    Implementation of the LLM abstract base class for the Mistral model.

    This class provides methods for initializing, loading, and interacting with the Mistral model.
    It includes support for custom system prompts and user roles.

    Attributes:
        model_name (str): The name of the Mistral model.
        role (str): The role of the user in the chat (default is 'user').
        system_prompt (str): The system prompt to guide the model's behavior.
    """

    def __init__(
        self,
        model_name: str,
        system_prompt: Optional[str] = None,
        system_prompt_file: Optional[str] = None,
        role: str = "user",
    ) -> None:
        """
        Initializes an MistralModel instance.

        Args:
            model_name (str): The name of the Mistral model to be loaded.
            system_prompt (Optional[str]): System prompt. Defaults to None.
            system_prompt_file (Optional[str]): Path to a file containing a custom system prompt. Defaults to None.
            role (str): The role of the user in the chat (e.g., 'user', 'assistant'). Defaults to 'user'.
        """
        self.api_key = Settings.MISTRAL_API_KEY
        super().__init__(model_name)
        logging.info(f"Using Mistral with {model_name} model 🤖")
        self.role: str = role
        self.system_prompt: str = ""
        if system_prompt_file is not None:
            self.system_prompt = self.load_system_prompt(system_prompt_file)
        elif system_prompt is not None:
            self.system_prompt = system_prompt
        else:
            self.system_prompt = Settings.DEFAULT_SYSTEM_PROMPT

    @override
    def load(self) -> Mistral:
        """
        Loads the Mistral model client.

        Returns:
            Client: An instance of the Mistral model client, configured with the necessary host and headers.
        """
        return Mistral(api_key=self.api_key)

    @override
    def generate(self, input: Dict[str, Any]) -> str:
        """
        Generates text using the Mistral model.

        Args:
            input (Dict[str, Any]): A dictionary containing the input data for text generation. The structure should
                                    include the necessary keys for the Mistral API.

        Returns:
            str: The generated output from the model.
        """
        input["system prompt"] = self.system_prompt
        new_input = dumps(input)
        response = self.model.chat.complete(
            model=self.model_name,
            messages=[
                {
                    "role": self.role,
                    "content": new_input,
                },
            ],
        )
        return response.choices[0].message.content

    @staticmethod
    def load_system_prompt(filePath: str) -> str:
        """
        Loads a custom system prompt from a file.

        Args:
            filePath (str): Path to the file containing the system prompt.

        Returns:
            str: The content of the system prompt file.
        """
        with open(filePath, "r", encoding="utf-8") as file:
            prompt = file.read()
        return prompt
