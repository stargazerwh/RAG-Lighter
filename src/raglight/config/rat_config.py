from dataclasses import dataclass, field
from ..llm.llm import LLM
from .rag_config import RAGConfig


@dataclass(kw_only=True)
class RATConfig(RAGConfig):
    reasoning_llm: LLM
    reflection: int = field(default=1)

    def get_rag_config(self):
        return RAGConfig(
            embedding_model=self.embedding_model,
            vector_store=self.vector_store,
            llm=self.llm,
            k=self.k,
        )
