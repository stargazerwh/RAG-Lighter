from __future__ import annotations
from typing import List
from typing_extensions import override

from sentence_transformers import SentenceTransformer

from .embeddings_model import EmbeddingsModel


class BgeM3EmbeddingsModel(EmbeddingsModel):
    """
    Concrete implementation for BGE-M3 embedding model.
    
    BGE-M3 is a multilingual embedding model that supports:
    - Dense retrieval (1024 dimensions)
    - Sparse retrieval (lexical matching)
    - Multi-vector retrieval (ColBERT-style)
    
    Model: BAAI/bge-m3 or BAAI/bge-large-en-v1.5
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        use_fp16: bool = True,
        device: str = None,
    ) -> None:
        """
        Initializes a BgeM3EmbeddingsModel instance.

        Args:
            model_name: Model name (default: BAAI/bge-m3)
            use_fp16: Use half precision for faster inference
            device: Device to use (cuda/cpu), auto-detected if None
        """
        super().__init__(model_name)
        self.use_fp16 = use_fp16
        self.device = device
        self.dim = 1024  # BGE-M3 dense dimension

    @override
    def load(self) -> SentenceTransformer:
        """
        Loads the BGE-M3 model.

        Returns:
            SentenceTransformer: The loaded model.
        """
        model = SentenceTransformer(self.model_name)
        
        if self.device:
            model = model.to(self.device)
        
        # Enable FP16 if requested and CUDA available
        if self.use_fp16 and model.device.type == "cuda":
            model.half()
            
        return model

    @override
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed list of documents using BGE-M3.
        
        For best results with BGE-M3, add instruction prefix:
        "Represent this sentence for searching relevant passages:"
        """
        # BGE-M3 works best with instruction for retrieval
        instruction = "Represent this sentence for searching relevant passages: "
        instructed_texts = [instruction + text for text in texts]
        
        embeddings = self.model.encode(
            instructed_texts,
            normalize_embeddings=True,  # Normalize for cosine similarity
            show_progress_bar=len(texts) > 10,
        )
        return embeddings.tolist()

    @override
    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query text.
        
        For best results with BGE-M3, add instruction prefix:
        "Represent this question for searching relevant passages:"
        """
        # BGE-M3 works best with instruction for retrieval
        instruction = "Represent this question for searching relevant passages: "
        instructed_text = instruction + text
        
        embedding = self.model.encode(
            instructed_text,
            normalize_embeddings=True,  # Normalize for cosine similarity
        )
        return embedding.tolist()

    def embed_documents_sparse(self, texts: List[str]) -> List[dict]:
        """
        Generate sparse embeddings (lexical weights) for hybrid retrieval.
        
        Returns:
            List of dicts mapping token ids to weights
        """
        # This requires the model to be loaded with sparse support
        # For now, return empty - can be extended with BGE-M3 specific sparse encoding
        return [{} for _ in texts]
