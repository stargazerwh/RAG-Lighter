from dataclasses import dataclass, field

from ..config.settings import Settings


@dataclass(kw_only=True)
class VectorStoreConfig:
    embedding_model: str
    persist_directory: str
    provider: str = field(default=Settings.HUGGINGFACE)
    file_extension: str = field(default=Settings.DEFAULT_EXTENSIONS)
    collection_name: str = field(default=Settings.DEFAULT_COLLECTION_NAME)
