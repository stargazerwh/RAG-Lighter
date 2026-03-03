"""
Parent-Child RAG Implementation
支持父子切块的 RAG 管道 - 小块检索，大块生成
"""

from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
import logging

from ..rag.rag import RAG
from ..vectorstore.parent_child_chroma import ParentChildChromaVS

logger = logging.getLogger(__name__)


class ParentChildRAG(RAG):
    """
    父子切块 RAG 实现
    
    与标准 RAG 的区别：
    1. 使用 ParentChildChromaVS 作为向量存储
    2. 检索时获取父块（完整上下文）而非子块
    3. 保持所有其他 RAG 功能（cross-encoder, streaming 等）
    
    优势：
    - 检索更精准（小块匹配）
    - 生成更全面（大块上下文）
    - 避免"半截句子"问题
    """

    def __init__(
        self,
        embedding_model,
        vector_store: ParentChildChromaVS,
        llm,
        k: int = 3,  # 返回的父块数量
        cross_encoder_model=None,
        stream: bool = False,
    ):
        """
        初始化父子切块 RAG

        Args:
            embedding_model: 嵌入模型
            vector_store: ParentChildChromaVS 实例
            llm: 大语言模型
            k: 返回的父块数量（父块，不是子块！）
            cross_encoder_model: 重排序模型（可选）
            stream: 是否流式输出
        """
        # 验证 vector_store 类型
        if not isinstance(vector_store, ParentChildChromaVS):
            raise ValueError(
                "ParentChildRAG requires ParentChildChromaVS, "
                f"got {type(vector_store).__name__}"
            )
        
        super().__init__(
            embedding_model=embedding_model,
            vector_store=vector_store,
            llm=llm,
            k=k,
            cross_encoder_model=cross_encoder_model,
            stream=stream
        )
        
        logger.info(
            f"ParentChildRAG initialized with k={k} parents"
        )

    def _retrieve(self, state: Dict) -> Dict[str, Any]:
        """
        检索节点 - 使用父子切块检索
        
        与标准 RAG 不同：
        - 内部在子块中搜索
        - 返回的是父块（完整上下文）
        """
        question = state["question"]
        
        # 使用 ParentChildChromaVS 的 search 方法
        # 它会自动：子块检索 -> 收集 parent_ids -> 返回父块
        retrieved_parents = self.vector_store.search(
            query=question,
            k=self.k
        )
        
        logger.info(
            f"Retrieved {len(retrieved_parents)} parent chunks for query"
        )
        
        return {
            "context": retrieved_parents,
            "question": question
        }

    def ingest_parent_child(self, parent_child_dict: Dict[str, List[Document]]):
        """
        导入父子切块数据
        
        Args:
            parent_child_dict: {"parents": [...], "children": [...]}
        """
        if hasattr(self.vector_store, 'ingest_parent_child'):
            self.vector_store.ingest_parent_child(parent_child_dict)
        else:
            raise ValueError("Vector store does not support parent-child ingest")

    def get_stats(self) -> Dict:
        """获取父子切块统计信息"""
        if hasattr(self.vector_store, 'get_stats'):
            return self.vector_store.get_stats()
        return {}


class ParentChildRAGBuilder:
    """
    父子切块 RAG 的构建器
    
    简化 ParentChildRAG 的创建流程
    """
    
    def __init__(self):
        self.embedding_model = None
        self.vector_store = None
        self.llm = None
        self.k = 3
        self.cross_encoder = None
        self.stream = False
        
        # 父子切块参数
        self.parent_chunk_size = 2000
        self.parent_chunk_overlap = 200
        self.child_chunk_size = 400
        self.child_chunk_overlap = 50

    def with_embeddings(self, model):
        """设置嵌入模型"""
        self.embedding_model = model
        return self

    def with_llm(self, llm):
        """设置 LLM"""
        self.llm = llm
        return self

    def with_vector_store(
        self,
        collection_name: str,
        persist_directory: str = "./parent_child_db",
        k_children: int = 10,
        k_parents: int = 3
    ):
        """
        配置父子切块向量存储
        
        Args:
            collection_name: 集合名称前缀
            persist_directory: 持久化目录
            k_children: 检索多少个子块
            k_parents: 返回多少个父块
        """
        from ..config.settings import Settings
        
        if self.embedding_model is None:
            raise ValueError("Must set embeddings before vector store")
        
        self.vector_store = ParentChildChromaVS(
            children_collection_name=collection_name,
            parents_collection_name=f"{collection_name}_parents",
            embeddings_model=self.embedding_model,
            persist_directory=persist_directory,
            k_children=k_children,
            k_parents=k_parents
        )
        
        self.k = k_parents
        return self

    def with_chunk_params(
        self,
        parent_size: int = 2000,
        parent_overlap: int = 200,
        child_size: int = 400,
        child_overlap: int = 50
    ):
        """配置父子切块参数"""
        self.parent_chunk_size = parent_size
        self.parent_chunk_overlap = parent_overlap
        self.child_chunk_size = child_size
        self.child_chunk_overlap = child_overlap
        return self

    def with_cross_encoder(self, cross_encoder):
        """设置重排序模型"""
        self.cross_encoder = cross_encoder
        return self

    def build(self) -> ParentChildRAG:
        """构建 ParentChildRAG 实例"""
        if not all([self.embedding_model, self.vector_store, self.llm]):
            raise ValueError("Missing required components")
        
        return ParentChildRAG(
            embedding_model=self.embedding_model,
            vector_store=self.vector_store,
            llm=self.llm,
            k=self.k,
            cross_encoder_model=self.cross_encoder,
            stream=self.stream
        )

    def build_with_processor(self):
        """
        构建 RAG 并返回配套 Processor
        
        Returns:
            (ParentChildRAG, ParentChildProcessor) 元组
        """
        from ..document_processing.parent_child_processor import ParentChildProcessor
        
        rag = self.build()
        
        processor = ParentChildProcessor(
            parent_chunk_size=self.parent_chunk_size,
            parent_chunk_overlap=self.parent_chunk_overlap,
            child_chunk_size=self.child_chunk_size,
            child_chunk_overlap=self.child_chunk_overlap
        )
        
        return rag, processor
