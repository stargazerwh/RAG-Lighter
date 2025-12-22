from __future__ import annotations
from typing import Iterable, Mapping, Optional, Dict, Any
from typing_extensions import override
from ..config.settings import Settings
from .llm import LLM
from ollama import Client
from json import dumps
import logging

# https://docs.ollama.com/context-length
OLLAMA_DEFAULT_CONTEXT_SIZE = 4096
OLLAMA_OPTION_CONTEXT_SIZE = "num_ctx"
OLLAMA_WARNING_CONTEXT_SIZE = 0.80


class OllamaModel(LLM):
    """
    Implementation of the LLM abstract base class for the Ollama model.

    This class provides methods for initializing, loading, and interacting with the Ollama model.
    It includes support for custom system prompts and user roles.

    Attributes:
        model_name (str): The name of the Ollama model.
        role (str): The role of the user in the chat (default is 'user').
        system_prompt (str): The system prompt to guide the model's behavior.
    """

    def __init__(
        self,
        model_name: str,
        options: Optional[Dict] = None,
        system_prompt: Optional[str] = None,
        system_prompt_file: Optional[str] = None,
        preload_model: Optional[bool] = True,
        api_base: Optional[str] = None,
        role: str = "user",
        headers: Optional[Mapping[str, str]] = None,
    ) -> None:
        """
        Initializes an OllamaModel instance.

        Args:
            model_name (str): The name of the Ollama model to be loaded.
            options (Optional[Dict]): Ollama options, both load and runtime, see https://github.com/ollama/ollama/blob/main/docs/modelfile.md#parameter
            system_prompt (Optional[str]): System prompt. Defaults to None.
            system_prompt_file (Optional[str]): Path to a file containing a custom system prompt. Defaults to None.
            role (str): The role of the user in the chat (e.g., 'user', 'assistant'). Defaults to 'user'.
            headers (Optional[Dict[str, str]]): Headers to be sent with the request. Defaults to None.
        """
        self.api_base = api_base or Settings.DEFAULT_OLLAMA_CLIENT
        self.headers = headers
        self.preload_model = preload_model
        self.options = options
        super().__init__(model_name, system_prompt, system_prompt_file, self.api_base)
        logging.info(f"Using Ollama with {model_name} model 🤖")
        self.role: str = role
        self.options = options
        self.max_context_size = (
            self.options.get(OLLAMA_OPTION_CONTEXT_SIZE, OLLAMA_DEFAULT_CONTEXT_SIZE)
            if self.options
            else OLLAMA_DEFAULT_CONTEXT_SIZE
        )

    @override
    def load(self) -> Client:
        """
        Loads the Ollama model client.

        Returns:
            Client: An instance of the Ollama model client, configured with the necessary host and headers.
        """
        ollama_client = Client(host=self.api_base, headers=self.headers)

        if self.preload_model:
            ollama_client.chat(
                model=self.model_name,
                messages=[],
                options=self.options,
            )
        return ollama_client

    @override
    def generate(self, input: Dict[str, Any]) -> str:
        """
        Generates text using the Ollama model.

        Args:
            input (Dict[str, Any]): A dictionary containing the input data for text generation. The structure should
                                    include the necessary keys for the Ollama API.

        Returns:
            str: The generated output from the model.
        """
        history = input.get("history", [])
        messages = []
        if len(history) > 1:
            messages.extend(history)

        user_prompt = input.get("question", "")
        user_message = {"role": self.role, "content": user_prompt}

        if "images" in input:
            images = [img["bytes"] for img in input["images"]]
            del input["images"]
            user_message["images"] = images
        messages.append(user_message)
        response = self.model.chat(
            model=self.model_name,
            messages=messages,
            options=self.options,
        )

        token_usage = response.eval_count + response.prompt_eval_count
        if token_usage / self.max_context_size > OLLAMA_WARNING_CONTEXT_SIZE:
            logging.warning(
                f"Over {OLLAMA_WARNING_CONTEXT_SIZE * 100}% of context window reached, consider increasing it or reducing prompt size."
                + f" Current usage : {token_usage}  out of {self.max_context_size} Tokens"
            )

        return response.message.content

    @override
    def generate_streaming(self, input: Dict[str, Any]) -> Iterable[str]:
        """
        Generates text using the Ollama model.

        Args:
            input (Dict[str, Any]): A dictionary containing the input data for text generation. The structure should
                                    include the necessary keys for the Ollama API.

        Yields:
              str: Chunks of the generated output as they become available.
        """
        input["system prompt"] = self.system_prompt
        new_input = dumps(input)
        response = self.model.chat(
            model=self.model_name,
            messages=[
                {
                    "role": self.role,
                    "content": new_input,
                },
            ],
            options=self.options,
            stream=True,
        )
        for chunk in response:
            yield chunk.message.content
