from __future__ import annotations

from ..cross_encoder.crossEncoderModel import CrossEncoderModel
from ..vectorestore.vectorStore import VectorStore
from ..embeddings.embeddingsModel import EmbeddingsModel
from ..llm.llm import LLM
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict, Dict
from langchain_core.documents import Document
from typing import Any


class State(TypedDict):
    """
    Represents the state of the RAG process.

    Attributes:
        question (str): The input question for the RAG process.
        context (List[Document]): A list of documents retrieved from the vector store as context.
        answer (str): The generated answer based on the input question and context.
    """

    question: str
    context: List[Document]
    answer: str


class RAG:
    """
    Implementation of a Retrieval-Augmented Generation (RAG) pipeline.

    This class integrates embeddings, a vector store, and a large language model (LLM) to
    retrieve relevant documents and generate answers based on a user's query.

    Attributes:
        embeddings: The embedding model used for vectorization.
        vector_store (VectorStore): The vector store instance for document retrieval.
        llm (LLM): The large language model instance for answer generation.
        k (int, optional): The number of top documents to retrieve. Defaults to 5.
        graph (StateGraph): The state graph that manages the RAG process flow.
    """

    def __init__(
        self,
        embedding_model: EmbeddingsModel,
        vector_store: VectorStore,
        llm: LLM,
        k: int,
        cross_encoder_model: CrossEncoderModel = None,
        stream: bool = False,
    ) -> None:
        """
        Initializes the RAG pipeline.

        Args:
            embedding_model (EmbeddingsModel): The embedding model used for vectorization.
            vector_store (VectorStore): The vector store for retrieving relevant documents.
            llm (LLM): The language model for generating answers.
        """
        self.embeddings: EmbeddingsModel = embedding_model.get_model()
        self.cross_encoder: CrossEncoderModel = (
            cross_encoder_model.get_model() if cross_encoder_model else None
        )
        self.vector_store: VectorStore = vector_store
        self.llm: LLM = llm
        self.k: int = k
        self.stream: bool = stream
        self.graph: Any = (
            self.createGraph()
        )  # Here type is CompiledGraph but it's not exposed by https://github.com/langchain-ai/langgraph/blob/main/libs/langgraph/langgraph/graph/graph.py

    def retrieve(self, state: Dict[str, str]) -> Dict[str, List[Document]]:
        """
        Retrieves relevant documents based on the input question.

        Args:
            state (Dict[str, str]): A dictionary containing the input question under the key 'question'.

        Returns:
            Dict[str, List[Document]]: A dictionary containing the retrieved documents under the key 'context'.
        """
        retrieved_docs = self.vector_store.similarity_search(
            state["question"], k=self.k
        )
        return {"context": retrieved_docs, "question": state["question"]}

    def generate_graph(self, state: Dict[str, List[Document]]) -> Dict[str, str]:
        """
        Generates an answer based on the input question and retrieved context.

        Args:
            state (Dict[str, List[Document]]): A dictionary containing:
                - 'question': The input question.
                - 'context': The list of retrieved documents.

        Returns:
            Dict[str, str]: A dictionary containing the generated answer under the key 'answer'.
        """
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        prompt_json = {"question": state["question"], "context": docs_content}
        if self.stream:
            response = self.llm.generate_streaming(prompt_json)
        else:
            response = self.llm.generate(prompt_json)
            return {"answer": response}

    def rerank(self, state: Dict[str, List[Document]]) -> Dict[str, List[Document]]:
        """
        Reranks the retrieved documents based on the cross-encoder model.

        Args:
            state (Dict[str, List[Document]]): A dictionary containing the list of retrieved documents under the key 'context'.

        Returns:
            Dict[str, List[Document]]: A dictionary containing the reranked documents under the key 'context'.
        """
        try:
            question = state["question"]
            docs = state["context"]
            doc_texts = [doc.page_content for doc in docs]
            scores = self.cross_encoder.predict(
                [(question, doc_text) for doc_text in doc_texts]
            )
            ranked_docs = [doc for _, doc in sorted(zip(scores, docs), reverse=True)]
        except:
            ranked_docs = state["context"]
        return {"context": ranked_docs}

    def createGraph(self) -> Any:
        """
        Creates and compiles the state graph for the RAG pipeline.

        Returns:
            StateGraph: The compiled state graph for managing the RAG process flow.
        """
        if self.cross_encoder:
            graph_builder = StateGraph(State).add_sequence(
                [self.retrieve, self.rerank, self.generate_graph]
            )
        else:
            graph_builder = StateGraph(State).add_sequence(
                [self.retrieve, self.generate_graph]
            )
        graph_builder.add_edge(START, "retrieve")
        return graph_builder.compile()

    def generate(self, question: str) -> str:
        """
        Executes the RAG pipeline for a given question.

        Args:
            question (str): The input question.

        Returns:
            str: The generated answer from the pipeline.
        """
        state = {"question": question}
        response = self.graph.invoke(state)
        return response["answer"]
