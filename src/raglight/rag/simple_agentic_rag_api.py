import asyncio
import logging
import shutil
import nest_asyncio
from typing import List

from ..config.vector_store_config import VectorStoreConfig
from .agentic_rag import AgenticRAG
from ..config.agentic_rag_config import AgenticRAGConfig
from ..vectorstore.vector_store import VectorStore
from ..scrapper.github_scrapper import GithubScrapper
from ..models.data_source_model import DataSource, FolderSource, GitHubSource

class AgenticRAGPipeline:
    def __init__(
        self,
        config: AgenticRAGConfig,
        vector_store_config: VectorStoreConfig,
    ) -> None:
        """
        Initializes the AgenticRAGPipeline.
        """
        self.config = config
        self.knowledge_base: List[DataSource] = config.knowledge_base
        self.ignore_folders = config.ignore_folders

        self.agenticRag = AgenticRAG(config, vector_store_config)

        self.github_scrapper: GithubScrapper = GithubScrapper()

    def build(self) -> None:
        """
        Builds the RAG pipeline by ingesting data from the knowledge base.

        This method processes the data sources (e.g., folders, GitHub repositories)
        and creates the embeddings for the vector store.
        """
        repositories: List[GitHubSource] = []
        if not self.knowledge_base:
            return
        for source in self.knowledge_base:
            if isinstance(source, FolderSource):
                self.get_vector_store().ingest(
                    data_path=source.path, ignore_folders=self.ignore_folders
                )
            if isinstance(source, GitHubSource):
                repositories.append(source)
        if len(repositories) > 0:
            self.ingest_github_repositories(repositories)

    def ingest_github_repositories(self, repositories: List[GitHubSource]) -> None:
        """
        Clones and processes GitHub repositories for the pipeline.

        Args:
            repositories (List[GitHubSource]): A list of GitHub repository sources to clone and ingest.
        """
        self.github_scrapper.set_repositories(repositories)
        repos_path: str = self.github_scrapper.clone_all()
        self.get_vector_store().ingest(
            data_path=repos_path, ignore_folders=self.ignore_folders
        )
        shutil.rmtree(repos_path)
        logging.info("✅ GitHub repositories cleaned successfully!")

    def get_vector_store(self) -> VectorStore:
        return self.agenticRag.vector_store

    def generate(self, question: str, stream: bool = False) -> str:
        """
        Synchronous wrapper for the agent's asynchronous generation.
        """
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        if loop.is_running():
            return loop.run_until_complete(self.agenticRag.generate(question, stream=stream))
        else:
            return asyncio.run(self.agenticRag.generate(question, stream=stream))