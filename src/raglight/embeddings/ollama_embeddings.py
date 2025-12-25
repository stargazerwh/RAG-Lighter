from __future__ import annotations
import logging
from typing import Optional, List, Dict, Any
from typing_extensions import override

from ..config.settings import Settings
from .embeddings_model import EmbeddingsModel
from ollama import Client


class OllamaEmbeddingsModel(EmbeddingsModel):
    """
    Concrete implementation of the EmbeddingsModel for Ollama models using the official python library.
    """

    def __init__(
        self,
        model_name: str,
        api_base: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> None:
        resolved_api_base = api_base or Settings.DEFAULT_OLLAMA_CLIENT
        super().__init__(model_name, api_base=resolved_api_base)

        self.options = options or {}

        # Keep critical config to prevent internal Ollama "panic" on large docs
        if "num_batch" not in self.options:
            self.options["num_batch"] = 8192
        if "num_ctx" not in self.options:
            self.options["num_ctx"] = 8192

    @override
    def load(self) -> Client:
        return Client(host=self.api_base)

    @override
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed list of documents using the optimized batch 'embed' method.
        """
        # OPTIMIZATION: Use 'embed' (not 'embeddings') to process the whole list at once.
        # This sends a single request and leverages GPU batch processing.
        response = self.model.embed(
            model=self.model_name, input=texts, options=self.options
        )
        return response["embeddings"]

    @override
    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query text.
        """
        response = self.model.embeddings(
            model=self.model_name, prompt=text, options=self.options
        )
        return response["embedding"]
