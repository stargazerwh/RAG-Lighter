import logging
import uuid
from typing import List, Dict, Optional, Any, cast
from typing_extensions import override

import chromadb
from chromadb.api.types import EmbeddingFunction, Documents, Embeddings
from langchain_core.documents import Document

from ..document_processing.document_processor import DocumentProcessor
from .vector_store import VectorStore
from ..embeddings.embeddings_model import EmbeddingsModel


class ChromaEmbeddingAdapter(EmbeddingFunction):
    """
    Adapter to make EmbeddingsModel compatible with ChromaDB's EmbeddingFunction interface.
    """

    def __init__(self, embeddings_model: EmbeddingsModel):
        self.embeddings_model = embeddings_model

    def __call__(self, input: Documents) -> Embeddings:
        if hasattr(self.embeddings_model, "embed_documents"):
            return self.embeddings_model.embed_documents(cast(List[str], input))
        else:
            raise TypeError(
                f"Object {type(self.embeddings_model)} does not implement 'embed_documents'."
            )


class ChromaVS(VectorStore):
    """
    Concrete implementation for ChromaDB using the official chromadb library.
    
    支持父子分块：
    - collection: 标准模式下的主集合 / 父子模式下的子块集合（用于检索）
    - collection_classes: 类签名集合
    - parent_collection: 父子模式下的父块集合（存储完整上下文）
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
        super().__init__(persist_directory, embeddings_model, custom_processors)

        self.persist_directory = persist_directory
        self.host = host
        self.port = port
        self.collection_name = collection_name

        self.embedding_function = ChromaEmbeddingAdapter(self.embeddings_model)

        if host and port:
            self.client = chromadb.HttpClient(host=host, port=port, ssl=False)
        elif persist_directory:
            self.client = chromadb.PersistentClient(path=persist_directory)
        else:
            raise ValueError(
                "Invalid configuration: provide host/port OR persist_directory."
            )

        # 主集合（标准模式：存储chunks；父子模式：存储children）
        self.collection = self.client.get_or_create_collection(
            name=collection_name, embedding_function=self.embedding_function
        )

        # 类签名集合
        self.collection_classes = self.client.get_or_create_collection(
            name=f"{collection_name}_classes",
            embedding_function=self.embedding_function,
        )
        
        # 父块集合（仅父子分块模式使用）
        self.parent_collection = None
        self._parent_collection_name = f"{collection_name}_parents"

    def _ensure_parent_collection(self):
        """确保父块集合已初始化"""
        if self.parent_collection is None:
            self.parent_collection = self.client.get_or_create_collection(
                name=self._parent_collection_name,
                embedding_function=self.embedding_function
            )
            logging.info(f"Initialized parent collection: {self._parent_collection_name}")

    @override
    def set_parent_child_config(self, use_parent_child: bool, chunk_config: Optional[Dict] = None):
        """设置父子分块配置并初始化父集合"""
        super().set_parent_child_config(use_parent_child, chunk_config)
        if use_parent_child:
            self._ensure_parent_collection()

    @override
    def add_documents(self, documents: List[Document]) -> None:
        if not documents:
            return

        logging.info(
            f"⏳ Adding {len(documents)} document chunks to ChromaDB collection '{self.collection.name}'..."
        )

        # Direct addition of all documents at once
        self._add_docs_to_collection(self.collection, documents)

        logging.info("✅ Documents successfully added to the main collection.")

    @override
    def add_class_documents(self, documents: List[Document]) -> None:
        if not documents:
            return

        logging.info(
            f"⏳ Adding {len(documents)} class documents to ChromaDB collection '{self.collection_classes.name}'..."
        )

        # Direct addition of all documents at once
        self._add_docs_to_collection(self.collection_classes, documents)

        logging.info("✅ Class documents successfully added to the class collection.")

    @override
    def add_parent_documents(self, documents: List[Document]) -> None:
        """添加父块到父集合"""
        if not documents:
            return
        
        self._ensure_parent_collection()
        
        logging.info(
            f"⏳ Adding {len(documents)} parent documents to collection '{self._parent_collection_name}'..."
        )
        
        # 使用 doc_id 作为 id（如果存在）
        ids = []
        for doc in documents:
            doc_id = doc.metadata.get("doc_id")
            if not doc_id:
                doc_id = str(uuid.uuid4())
                doc.metadata["doc_id"] = doc_id
            ids.append(doc_id)
        
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata if isinstance(doc.metadata, dict) else {} for doc in documents]
        
        self.parent_collection.add(ids=ids, documents=texts, metadatas=metadatas)
        logging.info("✅ Parent documents successfully added.")

    @override
    def add_child_documents(self, documents: List[Document]) -> None:
        """添加子块到主集合（用于检索）"""
        if not documents:
            return

        logging.info(
            f"⏳ Adding {len(documents)} child documents to ChromaDB collection '{self.collection.name}'..."
        )

        # 生成唯一 id
        ids = []
        for i, doc in enumerate(documents):
            parent_id = doc.metadata.get("parent_id", "unknown")
            child_id = f"child_{parent_id}_{i}_{uuid.uuid4().hex[:8]}"
            ids.append(child_id)

        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata if isinstance(doc.metadata, dict) else {} for doc in documents]

        self.collection.add(ids=ids, documents=texts, metadatas=metadatas)
        logging.info("✅ Child documents successfully added.")

    def _add_docs_to_collection(
        self, collection: Any, documents: List[Document]
    ) -> None:
        ids = [str(uuid.uuid4()) for _ in documents]
        texts = [doc.page_content for doc in documents]
        metadatas = [
            doc.metadata if isinstance(doc.metadata, dict) else {} for doc in documents
        ]

        collection.add(ids=ids, documents=texts, metadatas=metadatas)

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
            target_collection = self.client.get_or_create_collection(
                name=collection_name, embedding_function=self.embedding_function
            )

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
                target_collection = self.client.get_or_create_collection(
                    name=class_col_name, embedding_function=self.embedding_function
                )

        return self._query_collection(target_collection, question, k, filter)

    @override
    def similarity_search_parent_child(
        self,
        question: str,
        k: int = 5,
        filter: Optional[Dict[str, str]] = None,
        cross_encoder=None,
    ) -> List[Document]:
        """
        父子分块检索：
        1. 在子块集合中检索（多检索一些用于重排序）
        2. 【如果有cross_encoder】用 CrossEncoder 重排序子块
        3. 按顺序收集 parent_ids（去重）
        4. 从父块集合获取完整父块
        5. 返回父块列表
        
        Args:
            question: 查询问题
            k: 返回父块数量
            filter: 过滤条件
            cross_encoder: 可选的 CrossEncoder 模型，用于重排序子块
        """
        self._ensure_parent_collection()
        
        # Step 1: 检索更多子块（为重排序预留）
        retrieval_k = k * 10 if cross_encoder else k * 3
        child_results = self._query_collection_with_scores(
            self.collection, question, retrieval_k, filter
        )
        
        if not child_results:
            logging.warning("No child chunks found")
            return []
        
        # Step 2: 【如果有cross_encoder】重排序子块
        if cross_encoder:
            # 准备 (query, doc) pairs
            query_doc_pairs = [(question, r["content"]) for r in child_results]
            # CrossEncoder 预测分数
            rerank_scores = cross_encoder.predict(query_doc_pairs)
            
            # 按分数排序（降序）
            indexed_scores = list(enumerate(rerank_scores))
            indexed_scores.sort(key=lambda x: x[1], reverse=True)
            
            # 重新排列 child_results
            reranked_children = [child_results[idx] for idx, _ in indexed_scores]
            logging.info(f"Reranked {len(child_results)} children with CrossEncoder")
        else:
            reranked_children = child_results
        
        # Step 3: 按重排序后的子块顺序收集 parent_ids（去重，保留第一次出现的）
        seen_parents = set()
        parent_ids_ordered = []
        
        for result in reranked_children:
            parent_id = result["metadata"].get("parent_id")
            if parent_id and parent_id not in seen_parents:
                seen_parents.add(parent_id)
                parent_ids_ordered.append(parent_id)
        
        if not parent_ids_ordered:
            logging.warning("No parent IDs found in child results")
            return []
        
        # Step 4: 获取父块
        parent_results = self._get_documents_by_ids(
            self.parent_collection, parent_ids_ordered
        )
        
        # Step 5: 限制返回数量
        final_results = parent_results[:k]
        
        logging.info(
            f"Parent-Child search: {len(child_results)} children -> "
            f"reranked -> {len(parent_ids_ordered)} unique parents -> {len(final_results)} results"
        )
        
        return final_results

    def _query_collection(
        self, collection: Any, question: str, k: int, filter: Optional[Dict[str, Any]]
    ) -> List[Document]:
        results = collection.query(query_texts=[question], n_results=k, where=filter)

        found_docs: List[Document] = []
        if results["documents"] and results["documents"][0]:
            docs_list = results["documents"][0]
            metas_list = (
                results["metadatas"][0]
                if results["metadatas"]
                else [{}] * len(docs_list)
            )
            for text, meta in zip(docs_list, metas_list):
                safe_meta = meta if isinstance(meta, dict) else {}
                found_docs.append(Document(page_content=text, metadata=safe_meta))

        return found_docs

    def _query_collection_with_scores(
        self, collection: Any, question: str, k: int, filter: Optional[Dict[str, Any]]
    ) -> List[Dict]:
        """查询集合并返回带分数的结果"""
        results = collection.query(
            query_texts=[question], 
            n_results=k, 
            where=filter,
            include=["documents", "metadatas", "distances"]
        )

        found_results: List[Dict] = []
        if results["documents"] and results["documents"][0]:
            docs_list = results["documents"][0]
            metas_list = results["metadatas"][0] if results["metadatas"] else [{}] * len(docs_list)
            distances = results["distances"][0] if results["distances"] else [0] * len(docs_list)
            
            for text, meta, dist in zip(docs_list, metas_list, distances):
                found_results.append({
                    "content": text,
                    "metadata": meta if isinstance(meta, dict) else {},
                    "distance": dist
                })

        return found_results

    def _get_documents_by_ids(
        self, collection: Any, ids: List[str]
    ) -> List[Document]:
        """根据 ID 列表获取文档，保持传入顺序"""
        if not ids:
            return []
        
        results = collection.get(
            ids=ids,
            include=["documents", "metadatas"]
        )
        
        # 构建 id -> document 映射
        id_to_doc = {}
        for i, doc_id in enumerate(results["ids"]):
            doc = Document(
                page_content=results["documents"][i],
                metadata=results["metadatas"][i] if results["metadatas"] else {}
            )
            id_to_doc[doc_id] = doc
        
        # 按传入的 id 顺序返回
        ordered_docs = []
        for doc_id in ids:
            if doc_id in id_to_doc:
                ordered_docs.append(id_to_doc[doc_id])
        
        return ordered_docs

    @override
    def get_available_collections(self) -> List[str]:
        """
        Retrieves the list of available collections in the ChromaDB.
        """
        try:
            collections = self.client.list_collections()
            return [col.name for col in collections]
        except Exception as e:
            logging.error(f"Error listing collections: {e}")
            return []
