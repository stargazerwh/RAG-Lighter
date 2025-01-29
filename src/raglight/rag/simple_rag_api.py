from typing import List
import shutil
import logging
from ..rag.builder import Builder
from ..rag.rag import RAG
from ..vectorestore.vectorStore import VectorStore
from ..config.settings import Settings
from ..models.data_source_model import DataSource, FolderSource, GitHubSource
from ..scrapper.github_scrapper import GithubScrapper


class RAGPipeline:
    """
    A class that represents a Retrieval-Augmented Generation (RAG) pipeline.

    The pipeline combines various data sources (e.g., local folders, GitHub repositories),
    embeddings, and a language model to provide context-aware answers to questions.
    """

    def __init__(
        self,
        knowledge_base: List[DataSource],
        model_name: str = Settings.DEFAULT_LLM,
        provider: str = Settings.OLLAMA,
    ) -> None:
        """
        Initializes the RAGPipeline with a knowledge base and model.

        Args:
            knowledge_base (List[DataSource]): A list of data source objects (e.g., FolderSource, GitHubSource).
            model_name (str, optional): The name of the LLM to use. Defaults to Settings.DEFAULT_LLM.
        """
        self.knowledge_base: List[DataSource] = knowledge_base
        model_embeddings: str = Settings.DEFAULT_EMBEDDINGS_MODEL
        persist_directory: str = Settings.DEFAULT_PERSIST_DIRECTORY
        collection_name: str = Settings.DEFAULT_COLLECTION_NAME
        system_prompt: str = Settings.DEFAULT_SYSTEM_PROMPT
        self.file_extension: str = Settings.DEFAULT_EXTENSIONS
        self.rag: RAG = (
            Builder()
            .with_embeddings(Settings.HUGGINGFACE, model_name=model_embeddings)
            .with_vector_store(
                Settings.CHROMA,
                persist_directory=persist_directory,
                collection_name=collection_name,
            )
            .with_llm(provider, model_name=model_name, system_prompt=system_prompt)
            .build_rag()
        )
        self.github_scrapper: GithubScrapper = GithubScrapper()

    def get_vector_store(self) -> VectorStore:
        return self.rag.vector_store

    def build(self) -> None:
        """
        Builds the RAG pipeline by ingesting data from the knowledge base.

        This method processes the data sources (e.g., folders, GitHub repositories)
        and creates the embeddings for the vector store.
        """
        repositories: List[str] = []
        for source in self.knowledge_base:
            if isinstance(source, FolderSource):
                self.get_vector_store().ingest(
                    file_extension=self.file_extension, data_path=source.path
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
        self.get_vector_store().ingest_code(repos_path=repos_path)
        shutil.rmtree(repos_path)
        logging.info("✅ GitHub repositories cleaned successfully!")

    def generate(self, question: str) -> str:
        """
        Asks a question to the pipeline and retrieves the generated answer.

        Args:
            question (str): The question to ask the pipeline.

        Returns:
            str: The generated answer from the pipeline.
        """
        response: str = self.rag.question_graph(question)
        return response
