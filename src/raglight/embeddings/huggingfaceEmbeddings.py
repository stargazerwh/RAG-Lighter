from __future__ import annotations
from typing import Any
from typing_extensions import override
from .embeddingsModel import EmbeddingsModel
from langchain_huggingface import HuggingFaceEmbeddings


class HuggingfaceEmbeddingsModel(EmbeddingsModel):
    """
    Concrete implementation of the EmbeddingsModel for HuggingFace models.

    This class provides a specific implementation of the abstract `EmbeddingsModel` for
    loading and using HuggingFace embeddings.

    Attributes:
        model_name (str): The name of the HuggingFace model to be loaded.
    """

    def __init__(self, model_name: str) -> None:
        """
        Initializes a HuggingfaceEmbeddingsModel instance.

        Args:
            model_name (str): The name of the HuggingFace model to load.
        """
        super().__init__(model_name)

    @override
    def load(self) -> HuggingFaceEmbeddings:
        """
        Loads the HuggingFace embeddings model.

        This method overrides the abstract `load` method from the `EmbeddingsModel` class
        and initializes the HuggingFace embeddings model with the specified `model_name`.

        Returns:
            HuggingFaceEmbeddings: The loaded HuggingFace embeddings model.
        """
        return HuggingFaceEmbeddings(model_name=self.model_name)
