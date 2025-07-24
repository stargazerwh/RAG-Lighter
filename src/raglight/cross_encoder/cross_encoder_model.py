from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, List, Tuple


class CrossEncoderModel(ABC):
    """
    Abstract base class for cross encoder models.

    This class defines a blueprint for implementing cross encoder models with a consistent interface for
    loading and retrieving the model.

    Attributes:
        model_name (str): The name of the model.
        model (Any): The loaded model instance.
    """

    def __init__(self, model_name: str) -> None:
        """
        Initializes an CrossEncoderModel instance.

        Args:
            model_name (str): The name of the model to be loaded.
        """
        self.model_name: str = model_name
        self.model: Any = self.load()

    @abstractmethod
    def load(self) -> Any:
        """
        Abstract method to load the cross encoder model.

        This method must be implemented by any concrete subclass to define the loading process
        for the specific model.

        Returns:
            Any: The loaded model instance.
        """
        pass

    def get_model(self) -> CrossEncoderModel:
        """
        Retrieves the loaded cross encoder model.

        Returns:
            Any: The loaded model instance.
        """
        return self.model

    def predict(self, quer_list: List[Tuple[str, str]]) -> List[float]:
        """
        Abstract method to predict the similarity scores for a list of queries.

        Args:
            query_list (List[str]): A list of queries for which to predict the similarity scores.

        Returns:
            List[float]: The list of similarity scores for the input queries.
        """
        return self.model.predict(quer_list)
