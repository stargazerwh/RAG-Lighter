from ..rag.builder import Builder
from ..rag.rag import RAG
from ..config.settings import Settings
from typing import List
from ..models.data_source_model import DataSource, FolderSource


class RAGPipeline:
    def __init__(
        self, knowledge_base: List[DataSource], model_name=Settings.DEFAULT_LLM
    ) -> None:
        """
        Initialize the pipeline with data sources.
        :param data_sources: List of data source objects.
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
            .with_llm(
                Settings.OLLAMA, model_name=model_name, system_prompt=system_prompt
            )
            .build_rag()
        )

    def build(self) -> None:
        """
        Build the pipeline by creating embeddings and setting up the vector store.
        """
        for source in self.knowledge_base:
            if isinstance(source, FolderSource):
                self.rag.vector_store.ingest(
                    file_extension=self.file_extension, data_path=source.path
                )

    def generate(self, question: str) -> str:
        """
        Ask a question to the pipeline.
        :param question: The question to ask.
        :return: The generated answer.
        """
        response = self.rag.question_graph(question)
        return response
