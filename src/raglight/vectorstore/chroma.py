import logging
from typing import List, Dict, Optional
from typing_extensions import override
import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document

from ..document_processing.document_processor import DocumentProcessor

from .vector_store import VectorStore
from ..embeddings.embeddings_model import EmbeddingsModel


class ChromaVS(VectorStore):
    """
    Concrete implementation for ChromaDB.

    It inherits the main ingestion logic from the base VectorStore class and
    only implements the Chroma-specific methods for adding documents and
    performing searches.
    """

    def __init__(
        self,
        collection_name: str,
        embeddings_model: EmbeddingsModel,
        persist_directory: str = None,
        custom_processors: Optional[Dict[str, DocumentProcessor]] = None,
        host: str = None,
        port: int = None,
    ) -> None:
        """
        Initializes a ChromaVS instance.
        """
        super().__init__(persist_directory, embeddings_model, custom_processors)

        self.persist_directory = persist_directory
        self.host = host
        self.port = port

        if not host and not port:
            self.vector_store = Chroma(
                embedding_function=self.embeddings_model,
                persist_directory=persist_directory,
                collection_name=collection_name,
            )

            self.vector_store_classes = Chroma(
                embedding_function=self.embeddings_model,
                persist_directory=persist_directory,
                collection_name=f"{collection_name}_classes",
            )
        elif host and port:
            client = chromadb.HttpClient(host=host, port=port, ssl=False)
            self.vector_store = Chroma(
                client=client,
                embedding_function=self.embeddings_model,
                collection_name=collection_name,
            )

            self.vector_store_classes = Chroma(
                client=client,
                embedding_function=self.embeddings_model,
                collection_name=f"{collection_name}_classes",
            )
        else:
            raise ValueError(
                "Invalid configuration for ChromaVS: "
                "You must either:\n"
                "  • Provide both host and port (for remote ChromaDB), OR\n"
                "  • Provide a persist_directory (for local persistence).\n"
                f"Received -> host={host}, port={port}, persist_directory={persist_directory}"
            )

    @override
    def add_documents(self, documents: List[Document]) -> None:
        """
        Implements the logic to add documents specifically to the main ChromaDB collection,
        using batching for efficiency.
        """
        if not documents:
            return

        logging.info(
            f"⏳ Adding {len(documents)} document chunks to ChromaDB collection '{self.vector_store._collection_name}'..."
        )
        self.vector_store.add_documents(documents=documents)
        logging.info("✅ Documents successfully added to the main collection.")

    @override
    def add_class_documents(self, documents: List[Document]) -> None:
        """
        Implements the logic to add class signature documents to the dedicated ChromaDB
        collection for classes.
        """
        if not documents:
            return

        logging.info(
            f"⏳ Adding {len(documents)} class documents to ChromaDB collection '{self.vector_store_classes._collection_name}'..."
        )
        self.vector_store_classes.add_documents(documents=documents)
        logging.info("✅ Class documents successfully added to the class collection.")

    @override
    def similarity_search(
        self,
        question: str,
        k: int = 5,
        filter: Dict[str, str] = None,
        collection_name: str = None,
    ) -> List[Document]:
        """
        Implements similarity search using the main ChromaDB client.
        """
        if collection_name and collection_name != self.vector_store._collection_name:
            if not self.host and not self.port:
                vector_store = Chroma(
                    embedding_function=self.embeddings_model,
                    persist_directory=self.persist_directory,
                    collection_name=collection_name,
                )
            else:
                client = chromadb.HttpClient(host=self.host, port=self.port, ssl=False)
                vector_store = Chroma(
                    client=client,
                    embedding_function=self.embeddings_model,
                    collection_name=collection_name,
                )
            return vector_store.similarity_search(question, k=k, filter=filter)
        else:
            return self.vector_store.similarity_search(question, k=k, filter=filter)

    @override
    def similarity_search_class(
        self,
        question: str,
        k: int = 5,
        filter: Dict[str, str] = None,
        collection_name: str = None,
    ) -> List[Document]:
        """
        Implements similarity search using the dedicated class ChromaDB client.
        """
        if (
            collection_name
            and f"{collection_name}_classes"
            != self.vector_store_classes._collection_name
        ):
            if not self.host and not self.port:
                vector_store = Chroma(
                    embedding_function=self.embeddings_model,
                    persist_directory=self.persist_directory,
                    collection_name=f"{collection_name}_classes",
                )
            else:
                client = chromadb.HttpClient(host=self.host, port=self.port, ssl=False)
                vector_store = Chroma(
                    client=client,
                    embedding_function=self.embeddings_model,
                    collection_name=f"{collection_name}_classes",
                )
            return vector_store.similarity_search(question, k=k, filter=filter)
        return self.vector_store_classes.similarity_search(question, k=k, filter=filter)
