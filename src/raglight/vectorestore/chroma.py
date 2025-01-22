import logging
from typing import Any, List
from typing_extensions import override
from .vectorStore import VectorStore
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ..embeddings.embeddingsModel import EmbeddingsModel


class ChromaVS(VectorStore):
    """
    Concrete implementation of the VectorStore class using Chroma as the vector storage backend.

    This class handles the ingestion, indexing, and similarity search operations for document-based
    vector stores with Chroma as the underlying storage.

    Attributes:
        persist_directory (str): The directory where the Chroma index is persisted.
        vector_store (Chroma): The Chroma vector store instance.
        embeddings_model: The embeddings model used to encode the documents.
    """

    def __init__(
        self,
        persist_directory: str,
        collection_name: str,
        embeddings_model: EmbeddingsModel,
    ) -> None:
        """
        Initializes a ChromaVS instance.

        Args:
            persist_directory (str): Directory to persist the Chroma index.
            collection_name (str): Name of the collection within the Chroma vector store.
            embeddings_model (Any): The embeddings model used for vectorization.
        """
        super().__init__(persist_directory, embeddings_model)
        self.vector_store: Chroma = Chroma(
            embedding_function=self.embeddings_model,
            persist_directory=persist_directory,
            collection_name=collection_name,
        )
        self.persist_directory: str = persist_directory

    @override
    def ingest(self, **kwargs: Any) -> None:
        """
        Ingests documents, splits them into chunks, and indexes them in the vector store.

        Args:
            file_extension (str): File extension pattern (e.g., '*.txt') for document loading.
            data_path (str): Path to the directory containing the documents.
        """
        file_extension = kwargs.get("file_extension", "")
        data_path = kwargs.get("data_path", "")
        docs = self.load_docs(file_extension, data_path)
        all_splits = self.split_docs(docs)
        self.add_index(all_splits)
        logging.info("🎉 All documents ingested and indexed")

    @override
    def similarity_search(self, question: str, k: int = 2) -> List[Any]:
        """
        Performs a similarity search in the vector store.

        Args:
            question (str): The input query for similarity search.
            k (int, optional): The number of top results to retrieve. Defaults to 2.

        Returns:
            List[Any]: A list of top-k similar documents.
        """
        return self.vector_store.similarity_search(question, k=k)

    def split_docs(
        self, docs: List[Any], chunk_size: int = 2000, chunk_overlap: int = 100
    ) -> List[Any]:
        """
        Splits documents into smaller chunks for indexing.

        Args:
            docs (List[Any]): List of documents to be split.
            chunk_size (int, optional): Maximum size of each chunk. Defaults to 2000.
            chunk_overlap (int, optional): Overlap size between chunks. Defaults to 100.

        Returns:
            List[Any]: A list of document chunks.
        """
        logging.info("⏳ Splitting documents...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        all_splits = text_splitter.split_documents(docs)
        logging.info(f"✅ {len(all_splits)} document splits created")
        return all_splits

    def add_index(self, all_splits: List[Any]) -> None:
        """
        Adds document chunks to the Chroma vector store index.

        Args:
            all_splits (List[Any]): List of document chunks to index.

        Returns:
            Any: Result of the indexing operation.
        """
        logging.info("⏳ Adding documents to index...")
        _ = self.vector_store.add_documents(documents=all_splits)
        logging.info("✅ Documents added to index")

    def load_docs(self, file_extension: str, data_path: str) -> List[Any]:
        """
        Loads documents from a directory based on the specified file extension.

        Args:
            file_extension (str): File extension pattern (e.g., '*.txt') for loading documents.
            data_path (str): Path to the directory containing the documents.

        Returns:
            List[Any]: A list of loaded documents.
        """
        logging.info("⏳ Loading documents...")
        loader = DirectoryLoader(data_path, glob=file_extension)
        docs = loader.load()
        logging.info(f"✅ {len(docs)} documents loaded")
        return docs
