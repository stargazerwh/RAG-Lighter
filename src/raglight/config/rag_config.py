from dataclasses import dataclass, field
from typing import List, Optional

from ..config.settings import Settings
from ..cross_encoder.crossEncoderModel import CrossEncoderModel
from ..vectorestore.vectorStore import VectorStore
from ..models.data_source_model import DataSource


@dataclass(kw_only=True)
class RAGConfig:
    embedding_model: str
    cross_encoder_model: Optional[CrossEncoderModel] = None
    llm: str
    persist_directory: str
    provider: str = field(default=Settings.OLLAMA)
    file_extension: str = field(default=Settings.DEFAULT_EXTENSIONS)
    system_prompt: str = field(default=Settings.DEFAULT_SYSTEM_PROMPT)
    collection_name: str = field(default=Settings.DEFAULT_COLLECTION_NAME)
    k: int = field(default=2)
    stream: int = field(default=False)
    knowledge_base: List[DataSource] = field(default=None)
