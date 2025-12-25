from __future__ import annotations
from typing import List
from typing_extensions import override

from .embeddings_model import EmbeddingsModel
from langchain_huggingface import HuggingFaceEmbeddings


class HuggingfaceEmbeddingsModel(EmbeddingsModel):
    """
    Concrete implementation of the EmbeddingsModel for HuggingFace models.

    This class provides a specific implementation of the abstract `EmbeddingsModel` for
    loading and using HuggingFace embeddings via LangChain.
    """

    def __init__(self, model_name: str) -> None:
        """
        Initializes a HuggingfaceEmbeddingsModel instance.

        Args:
            model_name (str): The name of the HuggingFace model to load.
        """
        # Les modèles HuggingFace tournent généralement en local ou via le Hub sans base URL spécifique,
        # on laisse donc api_base à None via le constructeur parent.
        super().__init__(model_name)

    @override
    def load(self) -> HuggingFaceEmbeddings:
        """
        Loads the HuggingFace embeddings model via LangChain.

        Returns:
            HuggingFaceEmbeddings: The loaded HuggingFace embeddings model.
        """
        return HuggingFaceEmbeddings(model_name=self.model_name)

    @override
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed list of documents.

        Delegates to the underlying LangChain model's embed_documents method.
        """
        return self.model.embed_documents(texts)

    @override
    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query text.

        Delegates to the underlying LangChain model's embed_query method.
        """
        return self.model.embed_query(text)