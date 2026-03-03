from dataclasses import dataclass, field
from typing import List, Optional

from ..config.settings import Settings
from ..cross_encoder.cross_encoder_model import CrossEncoderModel
from ..models.data_source_model import DataSource


@dataclass(kw_only=True)
class RAGConfig:
    cross_encoder_model: Optional[CrossEncoderModel] = None
    api_base: str = field(default=Settings.DEFAULT_OLLAMA_CLIENT)
    llm: str
    provider: str = field(default=Settings.OLLAMA)
    system_prompt: str = field(default=Settings.DEFAULT_SYSTEM_PROMPT)
    k: int = field(default=2)
    stream: int = field(default=False)
    knowledge_base: List[DataSource] = field(default=None)
    ignore_folders: list = field(
        default_factory=lambda: list(Settings.DEFAULT_IGNORE_FOLDERS)
    )
    
    # === 父子分块配置 ===
    use_parent_child_chunking: bool = field(default=False)
    parent_chunk_size: int = field(default=2000)
    parent_chunk_overlap: int = field(default=200)
    child_chunk_size: int = field(default=400)
    child_chunk_overlap: int = field(default=50)
    
    # === 查询改写策略配置 ===
    # "Auto" = LLM动态选择, "Direct" = 直接查询, "HyDE" = 假设文档, "Subquery" = 子查询
    query_rewrite_strategy: str = field(default="Auto")
