"""
Parent-Child Vector Store Implementation
支持父子切块的向量存储 - 小块检索，大块生成
"""

import logging
from typing import List, Dict, Optional, Set
from langchain_core.documents import Document

from .chroma import ChromaVS
from ..embeddings.embeddings_model import EmbeddingsModel

logger = logging.getLogger(__name__)


class ParentChildChromaVS(ChromaVS):
    """
    支持父子切块的 ChromaDB 向量存储
    
    特点：
    1. 维护两个集合：children_collection（检索用）和 parents_collection（生成用）
    2. 检索时先在 children 中搜索，返回匹配的 parent
    3. 支持多种检索模式：child_only, parent_only, hierarchical
    
    Attributes:
        children_collection_name: 子块集合名称（用于向量检索）
        parents_collection_name: 父块集合名称（存储完整上下文）
    """

    def __init__(
        self,
        children_collection_name: str,
        parents_collection_name: str = None,
        embeddings_model: EmbeddingsModel = None,
        persist_directory: str = "./parent_child_db",
        k_children: int = 10,  # 检索的子块数量
        k_parents: int = 3,    # 返回的父块数量
    ):
        """
        初始化父子切块向量存储

        Args:
            children_collection_name: 子块集合名称
            parents_collection_name: 父块集合名称（默认为 children_name + "_parents"）
            embeddings_model: 嵌入模型
            persist_directory: 持久化目录
            k_children: 检索时取多少个子块
            k_parents: 最终返回多少个父块
        """
        # 父块集合名称默认自动生成
        if parents_collection_name is None:
            parents_collection_name = f"{children_collection_name}_parents"
        
        self.children_collection_name = children_collection_name
        self.parents_collection_name = parents_collection_name
        self.k_children = k_children
        self.k_parents = k_parents
        
        # 先初始化父集合（不依赖 embeddings）
        self.persist_directory = persist_directory
        self.embeddings_model = embeddings_model
        
        # 子集合用于检索（需要 embeddings）
        super().__init__(
            collection_name=children_collection_name,
            embeddings_model=embeddings_model,
            persist_directory=persist_directory
        )
        
        # 父集合用于存储完整内容（同样需要 embeddings，但不用于检索）
        # 实际上我们只需要存储，这里复用 Chroma 的存储能力
        self._init_parent_collection()
        
        logger.info(
            f"ParentChildChromaVS initialized:\n"
            f"  - Children collection: {children_collection_name} (for retrieval)\n"
            f"  - Parents collection: {parents_collection_name} (for context)\n"
            f"  - k_children={k_children}, k_parents={k_parents}"
        )

    def _init_parent_collection(self):
        """初始化父块集合"""
        try:
            import chromadb
            
            self.chroma_client = chromadb.PersistentClient(
                path=self.persist_directory
            )
            
            # 获取或创建父集合
            try:
                self.parent_collection = self.chroma_client.get_collection(
                    name=self.parents_collection_name
                )
                logger.info(f"Loaded existing parent collection: {self.parents_collection_name}")
            except Exception:
                self.parent_collection = self.chroma_client.create_collection(
                    name=self.parents_collection_name,
                    metadata={"hnsw:space": "cosine"}
                )
                logger.info(f"Created new parent collection: {self.parents_collection_name}")
                
        except Exception as e:
            logger.error(f"Failed to init parent collection: {e}")
            raise

    def ingest_parent_child(
        self, 
        parent_child_dict: Dict[str, List[Document]],
        batch_size: int = 100
    ):
        """
        批量导入父子切块

        Args:
            parent_child_dict: {"parents": [...], "children": [...]}
            batch_size: 批量大小
        """
        parents = parent_child_dict.get("parents", [])
        children = parent_child_dict.get("children", [])
        
        if not parents or not children:
            logger.warning("Empty parents or children, skipping ingest")
            return
        
        logger.info(f"Ingesting {len(parents)} parents, {len(children)} children")
        
        # Step 1: 导入父块（只需要存储，检索靠子块）
        self._ingest_parents(parents, batch_size)
        
        # Step 2: 导入子块（需要向量嵌入，用于检索）
        self._ingest_children(children, batch_size)
        
        logger.info("Parent-child ingest completed")

    def _ingest_parents(self, parents: List[Document], batch_size: int):
        """导入父块到父集合"""
        try:
            for i in range(0, len(parents), batch_size):
                batch = parents[i:i + batch_size]
                
                ids = [p.metadata.get("doc_id", f"parent_{i+j}") for j, p in enumerate(batch)]
                documents = [p.page_content for p in batch]
                metadatas = [p.metadata for p in batch]
                
                # 父块也需要 embedding 来存储（虽然不靠它检索）
                if self.embeddings_model:
                    embeddings = self.embeddings_model.embed_documents(documents)
                    self.parent_collection.add(
                        ids=ids,
                        documents=documents,
                        metadatas=metadatas,
                        embeddings=embeddings
                    )
                else:
                    self.parent_collection.add(
                        ids=ids,
                        documents=documents,
                        metadatas=metadatas
                    )
                
                logger.debug(f"Ingested {len(batch)} parents")
                
        except Exception as e:
            logger.error(f"Failed to ingest parents: {e}")
            raise

    def _ingest_children(self, children: List[Document], batch_size: int):
        """导入子块到子集合（使用父类的 ingest）"""
        # 子块需要生成 embedding 用于检索
        texts = [c.page_content for c in children]
        metadatas = [c.metadata for c in children]
        ids = [f"child_{c.metadata.get('parent_id', 'unknown')}_{i}" 
               for i, c in enumerate(children)]
        
        # 使用 embedding 模型生成向量
        if self.embeddings_model:
            logger.info(f"Generating embeddings for {len(children)} children...")
            embeddings = self.embeddings_model.embed_documents(texts)
        else:
            embeddings = None
        
        # 添加到 Chroma 集合
        for i in range(0, len(children), batch_size):
            batch_end = min(i + batch_size, len(children))
            
            batch_ids = ids[i:batch_end]
            batch_docs = texts[i:batch_end]
            batch_meta = metadatas[i:batch_end]
            batch_embeds = embeddings[i:batch_end] if embeddings else None
            
            self.collection.add(
                ids=batch_ids,
                documents=batch_docs,
                metadatas=batch_meta,
                embeddings=batch_embeds
            )
            
            logger.debug(f"Ingested children batch {i//batch_size + 1}")

    def search(
        self, 
        query: str, 
        k: int = None,
        filter_dict: Dict = None
    ) -> List[Document]:
        """
        父子切块检索
        
        流程：
        1. 在子块集合中检索（精准匹配）
        2. 收集匹配的 parent_ids
        3. 从父块集合中获取完整父块
        4. 去重并按相关性排序返回
        
        Args:
            query: 查询文本
            k: 返回父块数量（默认 self.k_parents）
            filter_dict: 过滤条件
            
        Returns:
            List[Document]: 父块列表（完整上下文）
        """
        if k is None:
            k = self.k_parents
        
        try:
            # Step 1: 在子块中检索
            child_results = self._search_children(query, self.k_children, filter_dict)
            
            if not child_results:
                logger.warning("No child chunks found")
                return []
            
            # Step 2: 收集 parent_ids（按匹配分数排序）
            parent_ids_ordered = self._collect_parent_ids(child_results)
            
            # Step 3: 获取父块
            parents = self._get_parents_by_ids(parent_ids_ordered)
            
            # Step 4: 限制返回数量
            final_parents = parents[:k]
            
            logger.info(
                f"Search: query='{query[:30]}...' "
                f"-> {len(child_results)} children "
                f"-> {len(parent_ids_ordered)} unique parents "
                f"-> return {len(final_parents)} parents"
            )
            
            return final_parents
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    def _search_children(
        self, 
        query: str, 
        k: int,
        filter_dict: Dict = None
    ) -> List[Dict]:
        """在子块集合中检索"""
        # 生成查询向量
        if self.embeddings_model:
            query_embedding = self.embeddings_model.embed_query(query)
        else:
            raise ValueError("Embeddings model required for search")
        
        # 执行检索
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where=filter_dict,
            include=["metadatas", "documents", "distances"]
        )
        
        # 解析结果
        child_results = []
        if results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                child_results.append({
                    "id": doc_id,
                    "content": results["documents"][0][i] if results["documents"] else "",
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "distance": results["distances"][0][i] if results["distances"] else 0
                })
        
        return child_results

    def _collect_parent_ids(self, child_results: List[Dict]) -> List[str]:
        """
        从子块结果中收集 parent_ids
        保持顺序（按匹配分数排序）并去重
        """
        seen = set()
        ordered_ids = []
        
        for child in child_results:
            parent_id = child["metadata"].get("parent_id")
            if parent_id and parent_id not in seen:
                seen.add(parent_id)
                ordered_ids.append(parent_id)
        
        return ordered_ids

    def _get_parents_by_ids(self, parent_ids: List[str]) -> List[Document]:
        """根据 ID 从父集合中获取父块"""
        if not parent_ids:
            return []
        
        try:
            results = self.parent_collection.get(
                ids=parent_ids,
                include=["documents", "metadatas"]
            )
            
            parents = []
            id_to_doc = {}
            
            for i, doc_id in enumerate(results["ids"]):
                doc = Document(
                    page_content=results["documents"][i],
                    metadata=results["metadatas"][i]
                )
                id_to_doc[doc_id] = doc
            
            # 保持传入的 ID 顺序
            for pid in parent_ids:
                if pid in id_to_doc:
                    parents.append(id_to_doc[pid])
            
            return parents
            
        except Exception as e:
            logger.error(f"Failed to get parents: {e}")
            return []

    def get_parent_by_child_id(self, child_id: str) -> Optional[Document]:
        """通过子块 ID 获取对应父块（单个查询）"""
        try:
            # 获取子块
            child_result = self.collection.get(
                ids=[child_id],
                include=["metadatas"]
            )
            
            if not child_result["ids"]:
                return None
            
            parent_id = child_result["metadatas"][0].get("parent_id")
            if not parent_id:
                return None
            
            # 获取父块
            parents = self._get_parents_by_ids([parent_id])
            return parents[0] if parents else None
            
        except Exception as e:
            logger.error(f"Failed to get parent by child id: {e}")
            return None

    def delete_parent_child(self, parent_id: str):
        """
        删除父块及其所有子块
        
        Args:
            parent_id: 父块 ID
        """
        try:
            # 1. 删除父块
            self.parent_collection.delete(ids=[parent_id])
            
            # 2. 删除所有关联的子块（通过 where 查询）
            self.collection.delete(
                where={"parent_id": parent_id}
            )
            
            logger.info(f"Deleted parent {parent_id} and its children")
            
        except Exception as e:
            logger.error(f"Failed to delete parent-child: {e}")

    def get_stats(self) -> Dict:
        """获取统计信息"""
        try:
            parent_count = self.parent_collection.count()
            child_count = self.collection.count()
            
            return {
                "parents_collection": self.parents_collection_name,
                "children_collection": self.children_collection_name,
                "parent_count": parent_count,
                "child_count": child_count,
                "ratio": child_count / parent_count if parent_count > 0 else 0
            }
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {}
