from typing import List
from typing_extensions import override

from ..config.vector_store_config import VectorStoreConfig
from .agentic_rag import AgenticRAG
from ..config.agentic_rag_config import AgenticRAGConfig
from ..vectorestore.vectorStore import VectorStore
from ..scrapper.github_scrapper import GithubScrapper
from ..config.settings import Settings
from ..models.data_source_model import DataSource
from .simple_rag_api import RAGPipeline


class AgenticRAGPipeline(RAGPipeline):
    def __init__(
        self,
        config: AgenticRAGConfig,
        vector_store_config: VectorStoreConfig,
    ) -> None:
        """
        Initializes the AgenticRAGPipeline with a knowledge base and model.

        Args:
            knowledge_base (List[DataSource]): A list of data source objects (e.g., FolderSource, GitHubSource).
            k (int, optional): The number of top documents to retrieve. Defaults to 5.
            model_name (str, optional): The name of the LLM to use. Defaults to Settings.DEFAULT_LLM.
            provider (str, optional): The name of the LLM provider you want to use : Ollama.
        """
        self.knowledge_base: List[DataSource] = config.knowledge_base
        self.file_extension: str = Settings.DEFAULT_EXTENSIONS

        self.agenticRag = AgenticRAG(config, vector_store_config)

        self.github_scrapper: GithubScrapper = GithubScrapper()

    @override
    def get_vector_store(self) -> VectorStore:
        return self.agenticRag.vector_store

    @override
    def generate(self, question: str, stream: bool = False) -> str:
        """
        Asks a question to the pipeline and retrieves the generated answer.

        Args:
            question (str): The question to ask the pipeline.

        Returns:
            str: The generated answer from the pipeline.
        """
        return self.agenticRag.generate(question, stream)
