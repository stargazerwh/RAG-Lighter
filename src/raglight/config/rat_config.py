from dataclasses import dataclass, field
from .settings import Settings
from .rag_config import RAGConfig


@dataclass(kw_only=True)
class RATConfig(RAGConfig):
    reasoning_llm: str = field(default=Settings.DEFAULT_REASONING_LLM)
    reflection: int = field(default=1)

    def get_rag_config(self):
        return RAGConfig(
            cross_encoder_model=self.cross_encoder_model,
            llm=self.llm,
            provider=self.provider,
            system_prompt=self.system_prompt,
            k=self.k,
            stream=self.stream,
            knowledge_base=self.knowledge_base,
        )
