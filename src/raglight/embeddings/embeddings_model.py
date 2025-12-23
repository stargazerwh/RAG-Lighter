from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Optional, List


class EmbeddingsModel(ABC):
    """
    Abstract base class for embeddings models.

    This class defines a blueprint for implementing embeddings models with a consistent interface for
    loading the model and generating embeddings for documents and queries.

    Attributes:
        model_name (str): The name of the model.
        model (Any): The loaded model instance (e.g., Ollama Client).
        api_base (Optional[str]): The base URL for the API.
    """

    def __init__(self, model_name: str, api_base: Optional[str] = None) -> None:
        """
        Initializes an EmbeddingsModel instance.

        Args:
            model_name (str): The name of the model to be loaded.
            api_base (Optional[str]): The base URL for the API (optional).
        """
        self.model_name: str = model_name
        self.api_base: Optional[str] = api_base
        # Load the model immediately upon initialization
        self.model: Any = self.load()

    @abstractmethod
    def load(self) -> Any:
        """
        Abstract method to load the embeddings model client.

        This method must be implemented by any concrete subclass to define the loading process
        for the specific model.

        Returns:
            Any: The loaded model instance.
        """
        pass

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Abstract method to embed a list of documents.

        Args:
            texts (List[str]): The list of texts to embed.

        Returns:
            List[List[float]]: List of embeddings, where each embedding is a list of floats.
        """
        pass

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """
        Abstract method to embed a single query text.

        Args:
            text (str): The text to embed.

        Returns:
            List[float]: The embedding as a list of floats.
        """
        pass

    def get_model(self) -> Any:
        """
        Retrieves the loaded embeddings model client.

        Returns:
            Any: The loaded model instance.
        """
        return self.model