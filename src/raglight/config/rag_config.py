from dataclasses import dataclass, field
from ..vectorestore.vectorStore import VectorStore
from ..embeddings.embeddingsModel import EmbeddingsModel
from ..llm.llm import LLM


@dataclass(kw_only=True)
class RAGConfig:
    embedding_model: EmbeddingsModel
    vector_store: VectorStore
    llm: LLM
    k: int = field(default=2)
