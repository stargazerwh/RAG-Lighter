from __future__ import annotations
from typing import Optional, List
from typing_extensions import override

from openai import OpenAI

from ..config.settings import Settings
from .embeddings_model import EmbeddingsModel


class OpenAIEmbeddingsModel(EmbeddingsModel):
    """
    Concrete implementation of the EmbeddingsModel for OpenAI models using the official python library.
    """

    def __init__(self, model_name: str, api_base: Optional[str] = None) -> None:
        """
        Initializes an OpenAIEmbeddingsModel instance.

        Args:
            model_name (str): The name of the OpenAI model to load.
            api_base (Optional[str]): The base URL for the API (optional).
        """
        resolved_api_base = api_base or Settings.DEFAULT_OPENAI_CLIENT
        super().__init__(model_name, api_base=resolved_api_base)

    @override
    def load(self) -> OpenAI:
        """
        Loads the OpenAI client.

        Returns:
            OpenAI: The initialized OpenAI client.
        """
        return OpenAI(
            api_key=Settings.OPENAI_API_KEY,
            base_url=self.api_base,
        )

    @override
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed list of documents using the official OpenAI client.
        """
        response = self.model.embeddings.create(input=texts, model=self.model_name)
        return [data.embedding for data in response.data]

    @override
    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query text.
        """
        response = self.model.embeddings.create(input=[text], model=self.model_name)
        return response.data[0].embedding
