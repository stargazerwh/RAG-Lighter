from raglight.rat.simple_rat_api import RATPipeline
from raglight.models.data_source_model import FolderSource, GitHubSource
from raglight.config.settings import Settings
from raglight.config.rat_config import RATConfig
from raglight.config.vector_store_config import VectorStoreConfig

Settings.setup_logging()

knowledge_base=[
    FolderSource(path="<path to the folder you want to ingest into your knowledge base>"),
    GitHubSource(url="https://github.com/Bessouat40/RAGLight")
    ]

vector_store_config = VectorStoreConfig(
    embedding_model = Settings.DEFAULT_EMBEDDINGS_MODEL,
    provider=Settings.HUGGINGFACE,
    database=Settings.CHROMA,
    persist_directory = './defaultDb',
    collection_name = Settings.DEFAULT_COLLECTION_NAME
)

config = RATConfig(
        cross_encoder_model = Settings.DEFAULT_CROSS_ENCODER_MODEL,
        llm = "llama3.2:3b",
        k = Settings.DEFAULT_K,
        provider = Settings.OLLAMA,
        system_prompt = Settings.DEFAULT_SYSTEM_PROMPT,
        reasoning_llm = Settings.DEFAULT_REASONING_LLM,
        reflection = 3
        # knowledge_base = knowledge_base,
    )

pipeline = RATPipeline(config, vector_store_config)

# This will ingest data from the knowledge base. Not mandatory if you have already ingested the data.
pipeline.build()

response = pipeline.generate("How can I create an easy RAGPipeline using raglight framework ? Give me the the easier python implementation")
print(response)
