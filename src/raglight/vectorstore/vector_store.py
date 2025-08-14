from abc import ABC, abstractmethod
from typing import Any, List, Dict
import os
import logging
from langchain_core.documents import Document
import copy

from ..document_processing.document_processor_factory import DocumentProcessorFactory
from ..embeddings.embeddings_model import EmbeddingsModel
from ..config.settings import Settings


class VectorStore(ABC):
    """
    Abstract base class for vector store implementations.

    This class provides a shared ingestion pipeline and defines the abstract methods
    that concrete implementations (like Chroma, Qdrant) must provide.
    """

    def __init__(
        self, persist_directory: str, embeddings_model: EmbeddingsModel
    ) -> None:
        """
        Initializes a VectorStore instance.
        """
        self.embeddings_model: Any = embeddings_model.get_model()
        self.persist_directory: str = persist_directory
        self.vector_store: Any = None
        self.vector_store_classes: Any = None

    def ingest(self, data_path: str, ignore_folders: List[str] = None) -> None:
        """
        Orchestrates the ingestion of documents by recursively walking the data_path,
        ignoring specified folders, and using a factory to select the correct
        processing strategy for each file. This logic is shared across all
        VectorStore implementations.
        """
        if not os.path.isdir(data_path):
            logging.error(f"Provided data_path '{data_path}' is not a valid directory.")
            return

        if ignore_folders is None:
            ignore_folders = Settings.DEFAULT_IGNORE_FOLDERS

        all_chunks = []
        all_class_docs = []
        factory = DocumentProcessorFactory()

        logging.info(f"⏳ Starting ingestion from '{data_path}'...")

        for root, dirs, files in os.walk(data_path, topdown=True):
            dirs[:] = [
                d
                for d in dirs
                if not self._should_ignore(os.path.join(root, d), ignore_folders)
            ]

            for file in files:
                file_path = os.path.join(root, file)
                processor = factory.get_processor(file_path)

                if processor:
                    logging.info(
                        f"  -> Processing '{file_path}' with {processor.__class__.__name__}"
                    )
                    try:
                        processed_docs = processor.process(
                            file_path, chunk_size=2500, chunk_overlap=250
                        )
                        all_chunks.extend(processed_docs.get("chunks", []))
                        all_class_docs.extend(processed_docs.get("classes", []))
                    except Exception as e:
                        logging.warning(f"⚠️ Error processing {file_path}: {e}")

        if not all_chunks and not all_class_docs:
            logging.warning(f"No processable documents were found in '{data_path}'.")
            return

        if all_chunks:
            all_chunks = self._flatten_metadata(all_chunks)
            self.add_documents(all_chunks)
        if all_class_docs:
            all_class_docs = self._flatten_metadata(all_class_docs)
            self.add_class_documents(all_class_docs)

        logging.info("🎉 Ingestion process completed successfully!")

    def _flatten_metadata(self, documents: List[Document]) -> List[Document]:
        """
        Creates a deep copy of the documents and converts any complex metadata values
        (lists, dicts) into their string representations.
        """
        cloned_documents = copy.deepcopy(documents)

        for doc in cloned_documents:
            for key, value in doc.metadata.items():
                if not isinstance(value, (str, int, float, bool)):
                    doc.metadata[key] = str(value)
        return cloned_documents

    @abstractmethod
    def add_documents(self, documents: List[Document]) -> None:
        """
        Adds a list of document chunks to the main vector store.
        """
        pass

    @abstractmethod
    def add_class_documents(self, documents: List[Document]) -> None:
        """
        Adds a list of class signature documents to the dedicated class vector store.
        """
        pass

    @abstractmethod
    def similarity_search(
        self, question: str, k: int = 5, filter: Dict[str, str] = None
    ) -> List[Document]:
        """
        Performs a similarity search in the main vector store.
        """
        pass

    @abstractmethod
    def similarity_search_class(
        self, question: str, k: int = 5, filter: Dict[str, str] = None
    ) -> List[Document]:
        """
        Performs a similarity search in the class vector store.
        """
        pass

    def _should_ignore(self, path: str, ignore_folders: List[str]) -> bool:
        """
        Checks if a given path should be ignored.
        """
        normalized_path = os.path.normpath(path)
        return any(folder in normalized_path.split(os.sep) for folder in ignore_folders)
