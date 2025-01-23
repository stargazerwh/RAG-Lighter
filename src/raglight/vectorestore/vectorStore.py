from abc import ABC, abstractmethod
from typing import Any, List
from ..embeddings.embeddingsModel import EmbeddingsModel


class VectorStore(ABC):
    """
    Abstract base class for vector store implementations.

    This class provides a blueprint for creating vector stores that handle document ingestion,
    indexing, and similarity search. Concrete implementations must define the methods for
    specific vector store backends.

    Attributes:
        persist_directory (str): The directory where the vector store data is persisted.
        embeddings_model: The embeddings model used for encoding documents.
        vector_store: The actual instance of the vector store, initialized in the subclass.
    """

    def __init__(
        self, persist_directory: str, embeddings_model: EmbeddingsModel
    ) -> None:
        """
        Initializes a VectorStore instance.

        Args:
            persist_directory (str): Directory where the vector store data will be persisted.
            embeddings_model (Any): The embeddings model instance used for vectorization.
        """
        self.embeddings_model: Any = embeddings_model.get_model()
        self.persist_directory: str = persist_directory
        self.vector_store: Any = None

    @abstractmethod
    def ingest(self, **kwargs: Any) -> None:
        """
        Abstract method to ingest and index documents in the vector store.

        Args:
            **kwargs (Any): Additional parameters required for ingestion, depending on the implementation.
        """
        pass

    @abstractmethod
    def ingest_code(self, **kwargs: Any) -> None:
        """
        Abstract method to ingest and index code in the vector store.

        Args:
            **kwargs (Any): Additional parameters required for ingestion, depending on the implementation.
        """
        pass

    @abstractmethod
    def similarity_search(self, question: str, k: int = 2) -> List[Any]:
        """
        Abstract method to perform similarity search in the vector store.

        Args:
            question (str): The input query for similarity search.
            k (int, optional): The number of top results to retrieve. Defaults to 2.

        Returns:
            List[Any]: A list of top-k similar documents.
        """
        pass
