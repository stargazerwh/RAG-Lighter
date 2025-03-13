from typing import List

from ..config.vector_store_config import VectorStoreConfig
from ..config.rat_config import RATConfig
from ..rag.builder import Builder
from ..rag.simple_rag_api import RAGPipeline
from .rat import RAT
from ..config.settings import Settings
from ..vectorestore.vectorStore import VectorStore
from ..models.data_source_model import DataSource
from ..scrapper.github_scrapper import GithubScrapper
from typing_extensions import override


class RATPipeline(RAGPipeline):
    """
    A pipeline for Retrieval-Augmented Thinking (RAT).

    This pipeline extends the Retrieval-Augmented Generation (RAG) concept by incorporating
    an additional reasoning step using a specialized reasoning language model (LLM). It combines
    various data sources (e.g., local folders, GitHub repositories), embeddings, and language models
    to provide both answers and reflections on user queries.
    """

    def __init__(
        self, config: RATConfig, vector_store_config: VectorStoreConfig
    ) -> None:
        """
        Initializes the RATPipeline with a knowledge base and models for answering and reasoning.

        Args:
            knowledge_base (List[DataSource]): A list of data sources (e.g., FolderSource, GitHubSource)
                to be used for document retrieval and context building.
            k (int, optional): The number of top documents to retrieve. Defaults to 5.
            model_name (str, optional): The name of the LLM to use for generating answers. Defaults to Settings.DEFAULT_LLM.
            reasoning_model_name (str, optional): The name of the LLM to use for reasoning. Defaults to Settings.DEFAULT_REASONING_LLM.
            reflection (int, optional): The number of reasoning iterations to perform. Defaults to 1.
        """
        self.knowledge_base: List[DataSource] = config.knowledge_base
        model_embeddings: str = vector_store_config.embedding_model
        persist_directory: str = vector_store_config.persist_directory
        collection_name: str = vector_store_config.collection_name
        database: str = vector_store_config.database
        embeddings_privider: str = vector_store_config.provider
        system_prompt: str = config.system_prompt
        self.file_extension: str = vector_store_config.file_extension
        self.rat: RAT = (
            Builder()
            .with_embeddings(embeddings_privider, model_name=model_embeddings)
            .with_vector_store(
                database,
                persist_directory=persist_directory,
                collection_name=collection_name,
            )
            .with_llm(
                config.provider, model_name=config.llm, system_prompt=system_prompt
            )
            .with_reasoning_llm(
                config.provider,
                model_name=config.reasoning_llm,
                system_prompt=system_prompt,
            )
            .build_rat(config.reflection, config.k)
        )
        self.github_scrapper: GithubScrapper = GithubScrapper()

    @override
    def get_vector_store(self) -> VectorStore:
        """
        Retrieves the vector store used in the pipeline.

        This method overrides the base RAGPipeline method to return the vector store
        configured within the RAT pipeline.

        Returns:
            VectorStore: The vector store instance used for document retrieval.
        """
        return self.rat.vector_store

    @override
    def generate(self, question: str) -> str:
        """
        Processes a question through the pipeline to retrieve both reasoning and a generated answer.

        This method first generates reasoning using the RAT pipeline and then produces a final answer
        that incorporates the reasoning.

        Args:
            question (str): The question to ask the pipeline.

        Returns:
            str: The generated answer from the pipeline, including reasoning.
        """
        response: str = self.rat.generate(question)
        return response
