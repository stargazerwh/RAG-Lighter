from dataclasses import dataclass, field
from ..llm.llm import LLM
from .rag_config import RAGConfig

@dataclass(kw_only=True)
class RATConfig(RAGConfig):
    reasoning_llm: LLM
    reflection: int = field(default=1)
