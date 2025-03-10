from raglight.rag.simple_rag_api import RAGPipeline
from raglight.models.data_source_model import FolderSource, GitHubSource
from raglight.config.settings import Settings
from raglight.config.rag_config import RAGConfig

Settings.setup_logging()

knowledge_base=[
    FolderSource(path="<path to your folder with pdf>/knowledge_base"),
    GitHubSource(url="https://github.com/Bessouat40/RAGLight")
    ],

config = RAGConfig(
        embedding_model = Settings.DEFAULT_EMBEDDINGS_MODEL,
        cross_encoder_model = Settings.DEFAULT_CROSS_ENCODER_MODEL,
        llm = Settings.DEFAULT_LLM,
        k = Settings.DEFAULT_K,
        persist_directory = './defaultDb',
        provider = Settings.OLLAMA,
        file_extension = Settings.DEFAULT_EXTENSIONS,
        system_prompt = Settings.DEFAULT_SYSTEM_PROMPT,
        collection_name = Settings.DEFAULT_COLLECTION_NAME,
        k = Settings.DEFAULT_K,
        knowledge_base = knowledge_base
    )

pipeline = RAGPipeline(config)

pipeline.build()

response = pipeline.generate("How can I create an easy RAGPipeline using raglight framework ? Give me python implementation")
print(response)
