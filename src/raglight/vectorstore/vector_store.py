from abc import ABC, abstractmethod
from typing import Any, List, Dict, Optional
import os
import logging
from langchain_core.documents import Document
import copy
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..document_processing.document_processor import DocumentProcessor
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
        self,
        persist_directory: str,
        embeddings_model: EmbeddingsModel,
        custom_processors: Optional[Dict[str, DocumentProcessor]] = None,
    ) -> None:
        """
        Initializes a VectorStore instance.
        """
        self.embeddings_model: EmbeddingsModel = embeddings_model
        self.persist_directory: str = persist_directory
        self.vector_store: Any = None
        self.vector_store_classes: Any = None
        self.custom_processors: Dict[str, DocumentProcessor] = custom_processors or {}
        
        # 父子分块相关
        self.use_parent_child: bool = False
        self.chunk_config: Optional[Dict] = None
        self.parent_store: Any = None  # 父块存储

    def set_parent_child_config(self, use_parent_child: bool, chunk_config: Optional[Dict] = None):
        """
        设置父子分块配置
        
        Args:
            use_parent_child: 是否启用父子分块
            chunk_config: 分块配置字典
        """
        self.use_parent_child = use_parent_child
        self.chunk_config = chunk_config or {}
        logging.info(f"VectorStore parent-child config: enabled={use_parent_child}")

    @staticmethod
    def _process_file(
        file_path: str, factory: DocumentProcessorFactory, flatten_metadata,
        use_parent_child: bool = False, chunk_config: Optional[Dict] = None
    ):
        processor = factory.get_processor(file_path)
        if not processor:
            return [], [], []
        try:
            processed_docs = processor.process(
                file_path, 
                chunk_size=2500, 
                chunk_overlap=250,
                use_parent_child=use_parent_child,
                chunk_config=chunk_config
            )
            
            if use_parent_child:
                # 父子分块模式
                parents = flatten_metadata(processed_docs.get("parents", []))
                children = flatten_metadata(processed_docs.get("children", []))
                classes = flatten_metadata(processed_docs.get("classes", []))
                return parents, children, classes
            else:
                # 标准模式
                chunks = flatten_metadata(processed_docs.get("chunks", []))
                classes = flatten_metadata(processed_docs.get("classes", []))
                return chunks, [], classes
                
        except Exception as e:
            logging.warning(f"⚠️ Error processing {file_path}: {e}")
            return [], [], []

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

        factory = DocumentProcessorFactory(custom_processors=self.custom_processors)

        logging.info(f"⏳ Starting ingestion from '{data_path}'...")
        if self.use_parent_child:
            logging.info("  Mode: Parent-Child Chunking")

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
                    self._process_file, file_path, factory, self._flatten_metadata,
                    self.use_parent_child, self.chunk_config
                )
                for file_path, _ in files_to_process
            ]

            for future in as_completed(futures):
                try:
                    if self.use_parent_child:
                        parents, children, classes = future.result()
                        if parents:
                            self.add_parent_documents(parents)
                        if children:
                            self.add_child_documents(children)
                        if classes:
                            self.add_class_documents(classes)
                    else:
                        chunks, _, classes = future.result()
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
    def add_parent_documents(self, documents: List[Document]) -> None:
        """
        Adds parent documents to the parent vector store (for parent-child chunking).
        """
        pass

    @abstractmethod
    def add_child_documents(self, documents: List[Document]) -> None:
        """
        Adds child documents to the child vector store (for parent-child chunking).
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

    @abstractmethod
    def similarity_search_parent_child(
        self, question: str, k: int = 5, filter: Dict[str, str] = None, cross_encoder=None
    ) -> List[Document]:
        """
        Performs a similarity search using parent-child chunking.
        Searches in child collection and returns parent documents.
        Optionally uses cross_encoder to rerank children before mapping to parents.
        """
        pass

    def _should_ignore(self, path: str, ignore_folders: List[str]) -> bool:
        """
        Checks if a given path should be ignored.
        """
        normalized_path = os.path.normpath(path)
        return any(folder in normalized_path.split(os.sep) for folder in ignore_folders)

    @abstractmethod
    def get_available_collections(self) -> List[str]:
        """
        Retrieves the list of available collections in the vector store.

        Returns:
            List[str]: A list of collection names available for querying.
        """
        pass
