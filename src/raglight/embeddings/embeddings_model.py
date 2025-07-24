from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any


class EmbeddingsModel(ABC):
    """
    Abstract base class for embeddings models.

    This class defines a blueprint for implementing embeddings models with a consistent interface for
    loading and retrieving the model.

    Attributes:
        model_name (str): The name of the model.
        model (Any): The loaded model instance.
    """

    def __init__(self, model_name: str) -> None:
        """
        Initializes an EmbeddingsModel instance.

        Args:
            model_name (str): The name of the model to be loaded.
        """
        self.model_name: str = model_name
        self.model: Any = self.load()

    @abstractmethod
    def load(self) -> Any:
        """
        Abstract method to load the embeddings model.

        This method must be implemented by any concrete subclass to define the loading process
        for the specific model.

        Returns:
            Any: The loaded model instance.
        """
        pass

    def get_model(self) -> EmbeddingsModel:
        """
        Retrieves the loaded embeddings model.

        Returns:
            Any: The loaded model instance.
        """
        return self.model
