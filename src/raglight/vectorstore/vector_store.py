from abc import ABC, abstractmethod
from typing import Any, List, Dict
import os
import logging
from langchain_core.documents import Document
import copy
from concurrent.futures import ThreadPoolExecutor, as_completed

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

    @staticmethod
    def _process_file(
        file_path: str, factory: DocumentProcessorFactory, flatten_metadata
    ):
        processor = factory.get_processor(file_path)
        if not processor:
            return [], []
        try:
            processed_docs = processor.process(
                file_path, chunk_size=2500, chunk_overlap=250
            )
            chunks = flatten_metadata(processed_docs.get("chunks", []))
            classes = flatten_metadata(processed_docs.get("classes", []))
            return chunks, classes
        except Exception as e:
            logging.warning(f"⚠️ Error processing {file_path}: {e}")
            return [], []

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

        factory = DocumentProcessorFactory()

        logging.info(f"⏳ Starting ingestion from '{data_path}'...")

        files_to_process = []
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
                        f"  -> Queuing '{file_path}' with {processor.__class__.__name__}"
                    )
                    files_to_process.append((file_path, processor))

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(
                    self._process_file, file_path, factory, self._flatten_metadata
                )
                for file_path, _ in files_to_process
            ]

            for future in as_completed(futures):
                try:
                    chunks, classes = future.result()
                    if chunks:
                        self.add_documents(chunks)
                    if classes:
                        self.add_class_documents(classes)
                except Exception as e:
                    logging.warning(f"⚠️ Future raised an exception: {e}")

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
