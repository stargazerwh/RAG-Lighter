from dataclasses import dataclass, field
from .settings import Settings
from .rag_config import RAGConfig


@dataclass(kw_only=True)
class RATConfig(RAGConfig):
    reasoning_llm: str = field(default=Settings.DEFAULT_REASONING_LLM)
    reflection: int = field(default=1)

    def get_rag_config(self):
        return RAGConfig(
            embedding_model=self.embedding_model,
            cross_encoder_model=self.cross_encoder_model,
            vector_store=self.vector_store,
            llm=self.llm,
            persist_directory=self.persist_directory,
            provider=self.provider,
            file_extension=self.file_extension,
            system_prompt=self.system_prompt,
            collection_name=self.collection_name,
            k=self.k,
            stream=self.stream,
            knowledge_base=self.knowledge_base,
        )
