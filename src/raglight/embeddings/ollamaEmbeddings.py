from __future__ import annotations
from typing import Any
from typing_extensions import override

from ..config.settings import Settings
from .embeddingsModel import EmbeddingsModel
from langchain_ollama import OllamaEmbeddings


class OllamaEmbeddingsModel(EmbeddingsModel):
    """
    Concrete implementation of the EmbeddingsModel for Ollama models.

    This class provides a specific implementation of the abstract `EmbeddingsModel` for
    loading and using Ollama embeddings.

    Attributes:
        model_name (str): The name of the Ollama model to be loaded.
    """

    def __init__(self, model_name: str) -> None:
        """
        Initializes a OllamaEmbeddingsModel instance.

        Args:
            model_name (str): The name of the Ollama model to load.
        """
        super().__init__(model_name)

    @override
    def load(self) -> OllamaEmbeddings:
        """
        Loads the Ollama embeddings model.

        This method overrides the abstract `load` method from the `EmbeddingsModel` class
        and initializes the Ollama embeddings model with the specified `model_name`.

        Returns:
            OllamaEmbeddings: The loaded Ollama embeddings model.
        """
        return OllamaEmbeddings(
            model=self.model_name, base_url=Settings.DEFAULT_OLLAMA_CLIENT
        )
