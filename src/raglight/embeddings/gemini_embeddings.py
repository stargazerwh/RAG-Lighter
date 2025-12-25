from __future__ import annotations
from typing import Optional, List
from typing_extensions import override

from ..config.settings import Settings
from .embeddings_model import EmbeddingsModel
from langchain_google_genai import GoogleGenerativeAIEmbeddings


class GeminiEmbeddingsModel(EmbeddingsModel):
    """
    Concrete implementation of the EmbeddingsModel for Gemini models.

    This class provides a specific implementation of the abstract `EmbeddingsModel` for
    loading and using Google Gemini embeddings via LangChain.
    """

    def __init__(self, model_name: str, api_base: Optional[str] = None) -> None:
        """
        Initializes a GeminiEmbeddingsModel instance.

        Args:
            model_name (str): The name of the Gemini model to load.
            api_base (Optional[str]): Base API config (optional).
        """
        # Logique inspirée de la classe Ollama : on résout l'api_base avant d'appeler super
        resolved_api_base = api_base or Settings.DEFAULT_GOOGLE_CLIENT
        super().__init__(model_name, api_base=resolved_api_base)

    @override
    def load(self) -> GoogleGenerativeAIEmbeddings:
        """
        Loads the Gemini embeddings model via LangChain.

        Returns:
            GoogleGenerativeAIEmbeddings: The loaded Gemini embeddings model.
        """
        return GoogleGenerativeAIEmbeddings(
            model=self.model_name, 
            google_api_key=Settings.GEMINI_API_KEY
        )

    @override
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed list of documents.
        
        Delegates to the underlying LangChain model's embed_documents method.
        LangChain typically handles the batching logic internally for Gemini.
        """
        return self.model.embed_documents(texts)

    @override
    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query text.
        
        Delegates to the underlying LangChain model's embed_query method.
        """
        return self.model.embed_query(text)