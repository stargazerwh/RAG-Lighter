from dataclasses import dataclass, field
from typing import Optional

from ..cross_encoder.crossEncoderModel import CrossEncoderModel
from ..vectorestore.vectorStore import VectorStore
from ..embeddings.embeddingsModel import EmbeddingsModel
from ..llm.llm import LLM


@dataclass(kw_only=True)
class RAGConfig:
    embedding_model: EmbeddingsModel
    cross_encoder_model: Optional[CrossEncoderModel] = None
    vector_store: VectorStore
    llm: LLM
    k: int = field(default=2)
    stream: int = field(default=False)
