from __future__ import annotations
from typing import Any, Optional, List
from typing_extensions import override

from ..config.settings import Settings
from .embeddings_model import EmbeddingsModel
from langchain_openai import OpenAIEmbeddings


class OpenAIEmbeddingsModel(EmbeddingsModel):
    """
    Concrete implementation of the EmbeddingsModel for OpenAI models.

    This class provides a specific implementation of the abstract `EmbeddingsModel` for
    loading and using OpenAI embeddings via LangChain.

    Attributes:
        model_name (str): The name of the OpenAI model to be loaded.
    """

    def __init__(self, model_name: str, api_base: Optional[str] = None) -> None:
        """
        Initializes an OpenAIEmbeddingsModel instance.

        Args:
            model_name (str): The name of the OpenAI model to load.
            api_base (Optional[str]): The base URL for the API (optional).
        """
        # Résolution de l'URL de base : soit celle fournie, soit la valeur par défaut définie dans Settings
        resolved_api_base = api_base or Settings.DEFAULT_OPENAI_CLIENT
        super().__init__(model_name, api_base=resolved_api_base)

    @override
    def load(self) -> OpenAIEmbeddings:
        """
        Loads the OpenAI embeddings model via LangChain.

        This method overrides the abstract `load` method from the `EmbeddingsModel` class
        and initializes the OpenAI embeddings model with the specified `model_name`.

        Returns:
            OpenAIEmbeddings: The loaded OpenAI embeddings model.
        """
        return OpenAIEmbeddings(
            model=self.model_name,
            openai_api_base=self.api_base,
            openai_api_key=Settings.OPENAI_API_KEY,
        )

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