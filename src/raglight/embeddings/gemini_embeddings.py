from __future__ import annotations
from typing import Optional
from typing_extensions import override

from ..config.settings import Settings
from .embeddings_model import EmbeddingsModel
from langchain_google_genai import GoogleGenerativeAIEmbeddings


class GeminiEmbeddingsModel(EmbeddingsModel):
    """
    Concrete implementation of the EmbeddingsModel for Gemini models.

    This class provides a specific implementation of the abstract `EmbeddingsModel` for
    loading and using Google Gemini embeddings.

    Attributes:
        model_name (str): The name of the Gemini model to be loaded.
    """

    def __init__(self, model_name: str, client_base: Optional[str] = None) -> None:
        """
        Initializes a GeminiEmbeddingsModel instance.

        Args:
            model_name (str): The name of the Gemini model to load.
        """
        self.client_base = client_base or Settings.DEFAULT_GOOGLE_CLIENT
        super().__init__(model_name)

    @override
    def load(self) -> GoogleGenerativeAIEmbeddings:
        """
        Loads the Gemini embeddings model.

        This method overrides the abstract `load` method from the `EmbeddingsModel` class
        and initializes the Google Generative AI embeddings model with the specified `model_name`.

        Returns:
            GoogleGenerativeAIEmbeddings: The loaded Gemini embeddings model.
        """
        return GoogleGenerativeAIEmbeddings(
            model=Settings.GEMINI_EMBEDDING_MODEL,
            google_api_base=self.client_base,
            google_api_key=Settings.GEMINI_API_KEY,
        )