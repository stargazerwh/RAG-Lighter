from __future__ import annotations
from typing import Any
from typing_extensions import override
from .cross_encoder_model import CrossEncoderModel
from sentence_transformers import CrossEncoder


class HuggingfaceCrossEncoderModel(CrossEncoderModel):
    """
    Concrete implementation of the CrossEncoderModel for HuggingFace models.

    This class provides a specific implementation of the abstract `CrossEncoderModel` for
    loading and using HuggingFace cross encoder.

    Attributes:
        model_name (str): The name of the HuggingFace model to be loaded.
    """

    def __init__(self, model_name: str) -> None:
        """
        Initializes a HuggingfaceCrossEncoderModel instance.

        Args:
            model_name (str): The name of the HuggingFace model to load.
        """
        super().__init__(model_name)

    @override
    def load(self) -> HuggingfaceCrossEncoderModel:
        """
        Loads the HuggingFace cross encoder model.

        This method overrides the abstract `load` method from the `CrossEncoderModel` class
        and initializes the HuggingFace cross encoder model with the specified `model_name`.

        Returns:
            HuggingfaceCrossEncoderModel: The loaded HuggingFace cross encoder model.
        """
        return CrossEncoder(self.model_name)
    
    @override
    def predict(self, query: str, documents: List[str], top_k: int) -> List[float]:
        """
        Abstract method to predict the similarity scores for a list of queries.

        Args:
            query_list (List[str]): A list of queries for which to predict the similarity scores.

        Returns:
            List[float]: The list of similarity scores for the input queries.
        """
        return self.model.rank(
            query = query,
            documents = documents,
            top_k = top_k,
            return_documents = True
            )
