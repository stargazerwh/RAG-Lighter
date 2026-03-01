from __future__ import annotations
import logging
import uuid
from typing import List, Dict, Optional, Any
from typing_extensions import override

from langchain_core.documents import Document
from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)

from ..document_processing.document_processor import DocumentProcessor
from .vector_store import VectorStore
from ..embeddings.embeddings_model import EmbeddingsModel


class MilvusVS(VectorStore):
    """
    Concrete implementation for Milvus vector database.
    Supports vector indexing for efficient similarity search.
    """

    def __init__(
        self,
        collection_name: str,
        embeddings_model: EmbeddingsModel,
        persist_directory: str = None,
        custom_processors: Optional[Dict[str, DocumentProcessor]] = None,
        host: str = "localhost",
        port: int = 19530,
        uri: str = None,
        token: str = None,
        index_type: str = "IVF_FLAT",
        metric_type: str = "COSINE",
    ) -> None:
        """
        Initialize Milvus vector store.

        Args:
            collection_name: Name of the collection
            embeddings_model: Model to generate embeddings
            persist_directory: Local path for Milvus Lite (optional)
            custom_processors: Custom document processors
            host: Milvus server host
            port: Milvus server port
            uri: Milvus connection URI (overrides host/port)
            token: Authentication token
            index_type: Vector index type (IVF_FLAT, HNSW, IVF_SQ8, etc.)
            metric_type: Distance metric (COSINE, L2, IP)
        """
        super().__init__(persist_directory, embeddings_model, custom_processors)

        self.collection_name = collection_name
        self.index_type = index_type
        self.metric_type = metric_type
        self.embeddings_model = embeddings_model

        # Get embedding dimension
        self.dim = self._get_embedding_dimension()

        # Connect to Milvus
        if uri:
            connections.connect(uri=uri, token=token)
        elif persist_directory:
            # Milvus Lite local mode
            connections.connect(uri=f"file:{persist_directory}/milvus.db")
        else:
            connections.connect(host=host, port=port, token=token)

        logging.info(f"✅ Connected to Milvus")

        # Initialize collections
        self.collection = self._get_or_create_collection(collection_name)
        self.collection_classes = self._get_or_create_collection(
            f"{collection_name}_classes"
        )

    def _get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings from the model."""
        try:
            # Try to get dimension from model
            if hasattr(self.embeddings_model, "dim"):
                return self.embeddings_model.dim
            # Test embedding
            test_embedding = self.embeddings_model.embed_query("test")
            return len(test_embedding)
        except Exception as e:
            logging.warning(f"Could not determine embedding dimension: {e}")
            return 768  # Default dimension

    def _get_or_create_collection(self, name: str) -> Collection:
        """Get existing collection or create new one with index."""
        if utility.has_collection(name):
            logging.info(f"📂 Using existing collection: {name}")
            collection = Collection(name)
            # Load collection for search
            collection.load()
            return collection

        logging.info(f"🆕 Creating new collection: {name}")

        # Define schema
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=36, is_primary=True),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dim),
            FieldSchema(name="metadata", dtype=DataType.JSON),
        ]

        schema = CollectionSchema(fields, description=f"RAG-Lighter collection: {name}")
        collection = Collection(name, schema)

        # Create index
        self._create_index(collection)

        # Load collection
        collection.load()

        return collection

    def _create_index(self, collection: Collection) -> None:
        """Create vector index for efficient search."""
        index_params = {
            "metric_type": self.metric_type,
            "index_type": self.index_type,
            "params": self._get_index_params(),
        }

        collection.create_index(field_name="embedding", index_params=index_params)
        logging.info(f"📊 Created {self.index_type} index with {self.metric_type} metric")

    def _get_index_params(self) -> Dict[str, Any]:
        """Get index-specific parameters."""
        params = {
            "IVF_FLAT": {"nlist": 128},
            "IVF_SQ8": {"nlist": 128},
            "HNSW": {"M": 16, "efConstruction": 200},
            "FLAT": {},
            "AUTOINDEX": {},
        }
        return params.get(self.index_type, {})

    @override
    def add_documents(self, documents: List[Document]) -> None:
        if not documents:
            return

        logging.info(
            f"⏳ Adding {len(documents)} document chunks to Milvus collection '{self.collection.name}'..."
        )

        self._add_docs_to_collection(self.collection, documents)
        logging.info("✅ Documents successfully added to the main collection.")

    @override
    def add_class_documents(self, documents: List[Document]) -> None:
        if not documents:
            return

        logging.info(
            f"⏳ Adding {len(documents)} class documents to Milvus collection '{self.collection_classes.name}'..."
        )

        self._add_docs_to_collection(self.collection_classes, documents)
        logging.info("✅ Class documents successfully added to the class collection.")

    def _add_docs_to_collection(
        self, collection: Collection, documents: List[Document]
    ) -> None:
        """Add documents to Milvus collection."""
        ids = [str(uuid.uuid4()) for _ in documents]
        texts = [doc.page_content for doc in documents]
        metadatas = [
            doc.metadata if isinstance(doc.metadata, dict) else {} for doc in documents
        ]

        # Generate embeddings
        logging.info(f"🔤 Generating embeddings for {len(documents)} documents...")
        embeddings = self.embeddings_model.embed_documents(texts)

        # Insert data
        entities = [
            ids,
            texts,
            embeddings,
            metadatas,
        ]

        collection.insert(entities)
        collection.flush()
        logging.info(f"💾 Inserted {len(documents)} entities")

    @override
    def similarity_search(
        self,
        question: str,
        k: int = 5,
        filter: Optional[Dict[str, str]] = None,
        collection_name: Optional[str] = None,
    ) -> List[Document]:
        target_collection = self.collection

        if collection_name and collection_name != self.collection.name:
            target_collection = Collection(collection_name)
            target_collection.load()

        return self._query_collection(target_collection, question, k, filter)

    @override
    def similarity_search_class(
        self,
        question: str,
        k: int = 5,
        filter: Optional[Dict[str, str]] = None,
        collection_name: Optional[str] = None,
    ) -> List[Document]:
        target_collection = self.collection_classes

        if collection_name:
            class_col_name = f"{collection_name}_classes"
            if class_col_name != self.collection_classes.name:
                target_collection = Collection(class_col_name)
                target_collection.load()

        return self._query_collection(target_collection, question, k, filter)

    def _query_collection(
        self,
        collection: Collection,
        question: str,
        k: int,
        filter: Optional[Dict[str, Any]],
    ) -> List[Document]:
        """Query Milvus collection."""
        # Generate query embedding
        query_embedding = self.embeddings_model.embed_query(question)

        # Build filter expression
        expr = None
        if filter:
            conditions = []
            for key, value in filter.items():
                conditions.append(f'metadata["{key}"] == "{value}"')
            expr = " and ".join(conditions)

        # Search parameters
        search_params = {
            "metric_type": self.metric_type,
            "params": {"ef": 64} if self.index_type == "HNSW" else {"nprobe": 10},
        }

        # Perform search
        results = collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=k,
            expr=expr,
            output_fields=["text", "metadata"],
        )

        # Convert to Documents
        found_docs: List[Document] = []
        for hits in results:
            for hit in hits:
                text = hit.entity.get("text")
                meta = hit.entity.get("metadata") or {}
                found_docs.append(Document(page_content=text, metadata=meta))

        return found_docs

    @override
    def get_available_collections(self) -> List[str]:
        """Retrieve list of available collections."""
        try:
            return utility.list_collections()
        except Exception as e:
            logging.error(f"Error listing collections: {e}")
            return []

    def __del__(self):
        """Cleanup connection on deletion."""
        try:
            connections.disconnect("default")
        except:
            pass
