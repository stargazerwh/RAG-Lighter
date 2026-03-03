from ..cross_encoder.cross_encoder_model import CrossEncoderModel
from ..vectorstore.vector_store import VectorStore
from ..embeddings.embeddings_model import EmbeddingsModel
from ..llm.llm import LLM
from ..config.rag_config import RAGConfig
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict, Dict, Optional
from langchain_core.documents import Document
from typing import Any
import logging

from .strategy_selector import StrategySelector
from .query_rewriter import QueryRewriter

logger = logging.getLogger(__name__)


class State(TypedDict):
    """
    Represents the state of the RAG process.

    Attributes:
        question (str): The input question for the RAG process.
        context (List[Document]): A list of documents retrieved from the vector store as context.
        answer (str): The generated answer based on the input question and context.
        history (List[Dict[str, str]]): The history of the conversation.
        strategy (str): The selected query rewrite strategy.
        queries (List[str]): The list of rewritten queries.
    """

    question: str
    answer: str
    context: List[Document]
    history: List[Dict[str, str]]
    strategy: str
    queries: List[str]


class RAG:
    """
    Implementation of a Retrieval-Augmented Generation (RAG) pipeline.
    
    增强功能：
    1. 查询策略选择：让 LLM 自动选择 Direct/HyDE/Subquery 策略
    2. 查询改写：根据策略改写查询
    3. 父子分块：支持小块检索，大块生成
    
    This class integrates embeddings, a vector store, and a large language model (LLM) to
    retrieve relevant documents and generate answers based on a user's query.

    Attributes:
        embeddings: The embedding model used for vectorization.
        vector_store (VectorStore): The vector store instance for document retrieval.
        llm (LLM): The large language model instance for answer generation.
        k (int, optional): The number of top documents to retrieve. Defaults to 5.
        graph (StateGraph): The state graph that manages the RAG process flow.
        config (RAGConfig): RAG 配置
        strategy_selector (StrategySelector): 策略选择器
        query_rewriter (QueryRewriter): 查询改写器
        use_parent_child (bool): 是否使用父子分块
    """

    def __init__(
        self,
        embedding_model: EmbeddingsModel,
        vector_store: VectorStore,
        llm: LLM,
        k: int = 5,
        cross_encoder_model: CrossEncoderModel = None,
        stream: bool = False,
        config: Optional[RAGConfig] = None,
    ) -> None:
        """
        Initializes the RAG pipeline.

        Args:
            embedding_model (EmbeddingsModel): The embedding model used for vectorization.
            vector_store (VectorStore): The vector store for retrieving relevant documents.
            llm (LLM): The language model for generating answers.
            k (int): The number of top documents to retrieve.
            cross_encoder_model: 重排序模型
            stream: 是否流式输出
            config (RAGConfig): RAG 配置，包含父子分块和查询改写配置
        """
        self.embeddings: EmbeddingsModel = embedding_model.get_model()
        self.cross_encoder: CrossEncoderModel = (
            cross_encoder_model if cross_encoder_model else None
        )
        self.vector_store: VectorStore = vector_store
        self.llm: LLM = llm
        self.k: int = k
        self.stream: bool = stream
        self.config: Optional[RAGConfig] = config
        
        # 初始化状态
        self.state: State = State(
            question="", 
            answer="", 
            context=[], 
            history=[],
            strategy="Direct",
            queries=[]
        )
        
        # 初始化策略选择器和查询改写器（如果配置启用）
        self.strategy_selector: Optional[StrategySelector] = None
        self.query_rewriter: Optional[QueryRewriter] = None
        
        if config:
            # 配置父子分块
            self.use_parent_child: bool = config.use_parent_child_chunking
            if self.use_parent_child:
                chunk_config = {
                    "parent_chunk_size": config.parent_chunk_size,
                    "parent_chunk_overlap": config.parent_chunk_overlap,
                    "child_chunk_size": config.child_chunk_size,
                    "child_chunk_overlap": config.child_chunk_overlap,
                }
                self.vector_store.set_parent_child_config(True, chunk_config)
                logger.info(f"Parent-child chunking enabled: {chunk_config}")
            
            # 配置查询改写（只要不是固定 Direct 都需要初始化）
            if config.query_rewrite_strategy != "Direct":
                self.strategy_selector = StrategySelector(llm)
                self.query_rewriter = QueryRewriter(llm)
                logger.info(f"Query rewrite enabled with strategy: {config.query_rewrite_strategy}")
        else:
            self.use_parent_child = False
        
        # 创建图
        self.graph: Any = self._createGraph()

    def _select_strategy(self, state: State) -> Dict[str, Any]:
        """
        选择查询策略
        
        如果配置为 "Auto"，让 LLM 动态选择；
        否则使用配置中固定的策略。
        """
        question = state["question"]
        
        if not self.config:
            return {"strategy": "Direct", "queries": [question], "question": question}
        
        strategy_config = self.config.query_rewrite_strategy
        
        if strategy_config == "Auto" and self.strategy_selector:
            # 让 LLM 动态选择策略
            selected_strategy = self.strategy_selector.select(question)
            logger.info(f"Auto-selected strategy: {selected_strategy}")
        else:
            # 使用固定策略
            selected_strategy = strategy_config
            logger.info(f"Using fixed strategy: {selected_strategy}")
        
        return {
            "strategy": selected_strategy,
            "question": question
        }

    def _rewrite_queries(self, state: State) -> Dict[str, Any]:
        """
        改写查询
        
        根据选定的策略改写查询：
        - Direct: [原查询]
        - HyDE: [假设文档, 原查询]
        - Subquery: [子查询1, 子查询2, ...]
        """
        question = state["question"]
        strategy = state.get("strategy", "Direct")
        
        if not self.query_rewriter:
            return {"queries": [question], "question": question}
        
        queries = self.query_rewriter.rewrite(question, strategy)
        logger.info(f"Rewrote query into {len(queries)} queries using {strategy}")
        
        return {
            "queries": queries,
            "question": question
        }

    def _retrieve(self, state: State) -> Dict[str, List[Document]]:
        """
        检索文档
        
        支持：
        1. 多查询检索（Subquery/HyDE 会产生多个查询）
        2. 父子分块检索（小块检索，返回父块）
        3. 标准检索
        
        所有结果按检索顺序直接拼接，不排序不截断。
        """
        queries = state.get("queries", [state["question"]])
        
        all_docs: List[Document] = []
        
        for query in queries:
            if self.use_parent_child:
                # 父子分块检索（传入 cross_encoder 用于子块重排序）
                docs = self.vector_store.similarity_search_parent_child(
                    query, k=self.k, cross_encoder=self.cross_encoder
                )
            else:
                # 标准检索
                docs = self.vector_store.similarity_search(query, k=self.k)
            
            all_docs.extend(docs)
            logger.debug(f"Query '{query[:50]}...' retrieved {len(docs)} docs")
        
        logger.info(f"Total retrieved: {len(all_docs)} documents from {len(queries)} queries")
        
        return {
            "context": all_docs,
            "question": state["question"]
        }

    def _rerank(self, state: State) -> Dict[str, List[Document]]:
        """
        重排序文档（可选）
        
        如果配置了 cross_encoder，对检索结果重排序。
        """
        if not self.cross_encoder:
            return state
        
        try:
            question = state["question"]
            docs = state["context"]
            
            if not docs:
                return state
            
            doc_texts = [doc.page_content for doc in docs]
            ranked_texts = self.cross_encoder.predict(
                question, doc_texts, int(self.k / 4)
            )
            
            ranked_docs = [Document(page_content=text) for text in ranked_texts]
            logger.info(f"Reranked {len(docs)} docs to {len(ranked_docs)}")
            
            return {
                "context": ranked_docs,
                "question": question
            }
            
        except Exception as e:
            logger.warning(f"Reranking failed: {e}")
            return state

    def _generate_graph(self, state: State) -> Dict[str, str]:
        """
        生成答案
        
        将所有检索到的上下文直接拼接（不排序不截断），生成最终答案。
        """
        docs = state["context"]
        question = state["question"]
        
        # 直接按检索顺序拼接上下文
        docs_content = "\n\n".join(doc.page_content for doc in docs)
        
        # 添加策略信息到 prompt（帮助调试）
        strategy = state.get("strategy", "Direct")
        queries = state.get("queries", [question])
        
        prompt = f"""Here is the retrieved context (excerpts from the document):
{docs_content}

Here is the question:
{question}

FINAL ANSWER (based only on the context):
"""
        
        if self.stream:
            response = self.llm.generate_streaming(
                {"question": prompt, "history": state.get("history", [])}
            )
            return {}  # Streaming returns nothing here
        else:
            response = self.llm.generate(
                {"question": prompt, "history": state.get("history", [])}
            )
            return {"answer": response}

    def _createGraph(self) -> Any:
        """
        创建 RAG 流程图
        
        流程：
        select_strategy -> rewrite_queries -> retrieve -> [rerank(标准模式)] -> generate
        
        注意：父子分块模式下，重排序在 similarity_search_parent_child 中完成（对子块重排序）
        """
        # 根据配置决定流程
        has_query_rewrite = self.config and self.config.query_rewrite_strategy != "Direct"
        # 父子分块模式下，重排序已在检索阶段完成
        needs_rerank_step = self.cross_encoder and not self.use_parent_child
        
        if has_query_rewrite and needs_rerank_step:
            # 完整流程：策略选择 -> 查询改写 -> 检索 -> 重排序 -> 生成
            graph_builder = StateGraph(State).add_sequence([
                self._select_strategy,
                self._rewrite_queries,
                self._retrieve,
                self._rerank,
                self._generate_graph
            ])
            
        elif has_query_rewrite:
            # 有查询改写，无重排序（或父子分块已包含重排序）
            graph_builder = StateGraph(State).add_sequence([
                self._select_strategy,
                self._rewrite_queries,
                self._retrieve,
                self._generate_graph
            ])
            
        elif needs_rerank_step:
            # 无查询改写，有重排序（标准模式）
            graph_builder = StateGraph(State).add_sequence([
                self._retrieve,
                self._rerank,
                self._generate_graph
            ])
            
        else:
            # 基础流程：检索 -> 生成
            graph_builder = StateGraph(State).add_sequence([
                self._retrieve,
                self._generate_graph
            ])
        
        graph_builder.add_edge(START, "_select_strategy" if has_query_rewrite else "_retrieve")
        return graph_builder.compile()

    def generate(self, question: str) -> str:
        """
        Executes the RAG pipeline for a given question.

        Args:
            question (str): The input question.

        Returns:
            str: The generated answer from the pipeline.
        """
        self.state["question"] = question
        self.state["context"] = []
        self.state["queries"] = []
        self.state["strategy"] = "Direct"
        
        response = self.graph.invoke(self.state)
        answer = response.get("answer", "")
        
        self.state["history"].extend([
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ])
        
        return answer

    def get_last_strategy(self) -> str:
        """获取上一次使用的策略（用于调试）"""
        return self.state.get("strategy", "Direct")
    
    def get_last_queries(self) -> List[str]:
        """获取上一次改写的查询列表（用于调试）"""
        return self.state.get("queries", [])
