from typing import List
from typing_extensions import override
import shutil
import logging

from ..config.vector_store_config import VectorStoreConfig
from .agentic_rag import AgenticRAG
from ..config.agentic_rag_config import AgenticRAGConfig
from ..vectorstore.vector_store import VectorStore
from ..scrapper.github_scrapper import GithubScrapper
from ..config.settings import Settings
from ..models.data_source_model import DataSource, FolderSource, GitHubSource
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
        self.ignore_folders = config.ignore_folders
        self.file_extension: str = Settings.DEFAULT_EXTENSIONS

        self.agenticRag = AgenticRAG(config, vector_store_config)

        self.github_scrapper: GithubScrapper = GithubScrapper()

    @override
    def get_vector_store(self) -> VectorStore:
        return self.agenticRag.vector_store

    @override
    def build(self) -> None:
        """
        Builds the AgenticRAG pipeline by ingesting data from the knowledge base.

        This method processes the data sources (e.g., folders, GitHub repositories)
        and creates the embeddings for the vector store.
        """
        repositories: List[str] = []
        if not self.knowledge_base:
            return
        for source in self.knowledge_base:
            if isinstance(source, FolderSource):
                self.get_vector_store().ingest(
                    file_extension=self.file_extension, 
                    data_path=source.path,
                    ignore_folders=self.ignore_folders
                )
            if isinstance(source, GitHubSource):
                repositories.append(source.url)
        if len(repositories) > 0:
            self.ingest_github_repositories(repositories)

    def ingest_github_repositories(self, repositories: List[str]) -> None:
        """
        Clones and processes GitHub repositories for the pipeline.

        Args:
            repositories (List[str]): A list of GitHub repository URLs to clone and ingest.
        """
        self.github_scrapper.set_repositories(repositories)
        repos_path: str = self.github_scrapper.clone_all()
        self.get_vector_store().ingest_code(repos_path=repos_path, ignore_folders=self.ignore_folders)
        shutil.rmtree(repos_path)
        logging.info("✅ GitHub repositories cleaned successfully!")

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
