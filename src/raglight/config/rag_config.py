from dataclasses import dataclass, field
from typing import List, Optional

from ..config.settings import Settings
from ..cross_encoder.crossEncoderModel import CrossEncoderModel
from ..models.data_source_model import DataSource


@dataclass(kw_only=True)
class RAGConfig:
    cross_encoder_model: Optional[CrossEncoderModel] = None
    llm: str
    provider: str = field(default=Settings.OLLAMA)
    system_prompt: str = field(default=Settings.DEFAULT_SYSTEM_PROMPT)
    k: int = field(default=2)
    stream: int = field(default=False)
    knowledge_base: List[DataSource] = field(default=None)
