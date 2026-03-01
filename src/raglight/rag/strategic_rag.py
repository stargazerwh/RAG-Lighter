from __future__ import annotations
from typing import List, Dict, Any, TypedDict, Literal
from enum import Enum
from langchain_core.documents import Document
import json
import logging

logger = logging.getLogger(__name__)


class RetrievalStrategy(Enum):
    """Enumeration of available retrieval strategies."""
    DIRECT = "direct"           # Direct vector search
    SUB_QUERY = "sub_query"     # Sub-query decomposition
    MULTI_HOP = "multi_hop"     # Multi-hop iterative retrieval
    HYBRID = "hybrid"           # Hybrid combination


class StrategyDecision(TypedDict):
    """Result of strategy selection."""
    strategy: str
    reasoning: str
    sub_queries: List[str]  # Sub-queries (for sub_query/multi_hop strategies)


class StrategicRAG:
    """
    RAG system with retrieval strategy selection.
    
    Pipeline: Query -> Strategy Selection -> Retrieval -> Generation
    """
    
    def __init__(
        self,
        llm,  # LLM for generation and strategy selection
        vector_store,
        embedding_model,
        strategy_llm=None,  # Dedicated LLM for strategy selection (optional)
    ):
        self.llm = llm
        self.strategy_llm = strategy_llm or llm
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        
    def select_strategy(self, query: str) -> StrategyDecision:
        """
        Use LLM to select the best retrieval strategy for the query.
        
        Args:
            query: User query string
            
        Returns:
            StrategyDecision: Contains strategy type, reasoning, and sub-queries
        """
        prompt = f"""Analyze the following user query and select the most appropriate retrieval strategy.

User Query: {query}

Available Strategies:
1. direct - Direct retrieval: For simple, straightforward queries
2. sub_query - Sub-query decomposition: For complex queries requiring multiple aspects
3. multi_hop - Multi-hop retrieval: For queries requiring multi-step reasoning
4. hybrid - Hybrid approach: Combines multiple strategies

Respond in JSON format:
{{
    "strategy": "direct|sub_query|multi_hop|hybrid",
    "reasoning": "Explanation for selecting this strategy",
    "sub_queries": ["sub-query 1", "sub-query 2"]  // Only for sub_query/multi_hop
}}

Response:"""
        
        try:
            response = self.strategy_llm.generate({"question": prompt})
            # Parse JSON response
            decision = json.loads(response.strip())
            return StrategyDecision(
                strategy=decision.get("strategy", "direct"),
                reasoning=decision.get("reasoning", ""),
                sub_queries=decision.get("sub_queries", [])
            )
        except Exception as e:
            logger.warning(f"Strategy selection failed, using default: {e}")
            return StrategyDecision(
                strategy="direct",
                reasoning="Default strategy due to error",
                sub_queries=[]
            )
    
    def retrieve_direct(self, query: str, k: int = 5) -> List[Document]:
        """Direct vector similarity search."""
        return self.vector_store.similarity_search(query, k=k)
    
    def retrieve_sub_query(
        self, 
        query: str, 
        sub_queries: List[str], 
        k: int = 3
    ) -> List[Document]:
        """
        Sub-query retrieval strategy.
        
        Retrieve for each sub-query and merge results with deduplication.
        """
        all_docs = []
        seen_content = set()
        
        # Retrieve for original query
        docs = self.vector_store.similarity_search(query, k=k)
        for doc in docs:
            if doc.page_content not in seen_content:
                all_docs.append(doc)
                seen_content.add(doc.page_content)
        
        # Retrieve for each sub-query
        for sub_q in sub_queries:
            docs = self.vector_store.similarity_search(sub_q, k=k)
            for doc in docs:
                if doc.page_content not in seen_content:
                    all_docs.append(doc)
                    seen_content.add(doc.page_content)
        
        return all_docs
    
    def retrieve_multi_hop(
        self, 
        query: str, 
        max_hops: int = 2
    ) -> List[Document]:
        """
        Multi-hop retrieval strategy.
        
        Iterative retrieval: Query -> Retrieve -> Generate intermediate -> New query -> ...
        """
        all_docs = []
        current_query = query
        
        for hop in range(max_hops):
            docs = self.vector_store.similarity_search(current_query, k=3)
            all_docs.extend(docs)
            
            if hop < max_hops - 1:
                # Generate intermediate answer, extract key info for next hop
                context = "\n".join([d.page_content for d in docs])
                prompt = f"""Based on the following information, extract key entities and relationships, then generate a more specific query:

Context: {context[:1000]}
Original Query: {query}

Generate a more specific query to retrieve additional information:"""
                
                current_query = self.llm.generate({"question": prompt})
                logger.info(f"Hop {hop+1} query: {current_query}")
        
        # Deduplicate
        seen = set()
        unique_docs = []
        for doc in all_docs:
            if doc.page_content not in seen:
                unique_docs.append(doc)
                seen.add(doc.page_content)
        
        return unique_docs
    
    def retrieve(self, query: str, k: int = 5) -> tuple[List[Document], StrategyDecision]:
        """
        Execute retrieval based on selected strategy.
        
        Returns:
            (list of documents, strategy decision)
        """
        # 1. Select strategy
        decision = self.select_strategy(query)
        logger.info(f"Selected strategy: {decision['strategy']}, Reason: {decision['reasoning']}")
        
        # 2. Execute corresponding retrieval
        strategy = decision["strategy"]
        
        if strategy == "sub_query" and decision["sub_queries"]:
            docs = self.retrieve_sub_query(query, decision["sub_queries"], k=k)
        elif strategy == "multi_hop":
            docs = self.retrieve_multi_hop(query, max_hops=2)
        elif strategy == "hybrid":
            # Hybrid: direct retrieval + sub-query retrieval
            docs_direct = self.retrieve_direct(query, k=k)
            docs_sub = self.retrieve_sub_query(
                query, 
                decision.get("sub_queries", [query]), 
                k=2
            )
            # Merge and deduplicate
            seen = set(d.page_content for d in docs_direct)
            docs = docs_direct + [d for d in docs_sub if d.page_content not in seen]
        else:
            # Default direct retrieval
            docs = self.retrieve_direct(query, k=k)
        
        return docs, decision
    
    def generate(
        self, 
        query: str, 
        docs: List[Document], 
        strategy_info: StrategyDecision
    ) -> Dict[str, Any]:
        """
        Generate final answer.
        
        Include strategy information in prompt to help LLM better utilize retrieval results.
        """
        context = "\n\n".join([
            f"[Document {i+1}] {doc.page_content}" 
            for i, doc in enumerate(docs)
        ])
        
        prompt = f"""Answer the question based on the following retrieved documents.

Retrieval Strategy: {strategy_info['strategy']}
Strategy Reasoning: {strategy_info['reasoning']}

Retrieved Documents:
{context}

User Question: {query}

Please answer based on the above documents. If the information is insufficient, please state so explicitly.

Answer:"""
        
        answer = self.llm.generate({"question": prompt})
        
        return {
            "answer": answer,
            "strategy": strategy_info["strategy"],
            "reasoning": strategy_info["reasoning"],
            "retrieved_docs": len(docs),
            "context": context,
        }
    
    def query(self, query: str) -> Dict[str, Any]:
        """
        Complete query pipeline: Strategy Selection -> Retrieval -> Generation
        
        Args:
            query: User query string
            
        Returns:
            Dictionary containing answer, strategy, and retrieval results
        """
        # 1. Retrieval (includes strategy selection)
        docs, strategy = self.retrieve(query)
        
        # 2. Generation
        result = self.generate(query, docs, strategy)
        
        return result


# ============ Usage Example ============

if __name__ == "__main__":
    from raglight.llm import OpenAIModel
    from raglight.vectorstore import ChromaVS
    from raglight.embeddings import HuggingfaceEmbeddingsModel
    
    # Initialize components
    llm = OpenAIModel(model_name="gpt-4")
    embeddings = HuggingfaceEmbeddingsModel("all-MiniLM-L6-v2")
    vector_store = ChromaVS(
        collection_name="my_docs",
        embeddings_model=embeddings,
        persist_directory="./db"
    )
    
    # Create strategic RAG
    rag = StrategicRAG(
        llm=llm,
        vector_store=vector_store,
        embedding_model=embeddings,
    )
    
    # Example queries
    queries = [
        "What is RAG?",  # Simple query -> direct
        "Compare the advantages and disadvantages of RAG and fine-tuning, and provide use cases",  # Complex -> sub_query
        "Find all papers related to Transformer and their citation relationships",  # Relational -> multi_hop
    ]
    
    for q in queries:
        print(f"\n{'='*50}")
        print(f"Query: {q}")
        result = rag.query(q)
        print(f"Strategy: {result['strategy']}")
        print(f"Reasoning: {result['reasoning']}")
        print(f"Retrieved Documents: {result['retrieved_docs']}")
        print(f"Answer: {result['answer'][:200]}...")
