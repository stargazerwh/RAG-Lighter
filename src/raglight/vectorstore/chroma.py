import logging
from typing import List, Dict
from typing_extensions import override
import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document

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
        host: str = None,
        port: int = None,
    ) -> None:
        """
        Initializes a ChromaVS instance.
        """
        super().__init__(persist_directory, embeddings_model)

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
        self, question: str, k: int = 5, filter: Dict[str, str] = None
    ) -> List[Document]:
        """
        Implements similarity search using the main ChromaDB client.
        """
        return self.vector_store.similarity_search(question, k=k, filter=filter)

    @override
    def similarity_search_class(
        self, question: str, k: int = 5, filter: Dict[str, str] = None
    ) -> List[Document]:
        """
        Implements similarity search using the dedicated class ChromaDB client.
        """
        return self.vector_store_classes.similarity_search(question, k=k, filter=filter)
