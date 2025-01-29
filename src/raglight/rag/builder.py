from __future__ import annotations
import logging
from typing import Optional
from ..llm.llm import LLM
from ..llm.ollamaModel import OllamaModel
from ..llm.lmStudioModel import LMStudioModel
from ..vectorestore.vectorStore import VectorStore
from ..vectorestore.chroma import ChromaVS
from ..config.settings import Settings
from .rag import RAG
from ..rat.rat import RAT
from ..embeddings.embeddingsModel import EmbeddingsModel
from ..embeddings.huggingfaceEmbeddings import HuggingfaceEmbeddingsModel


class Builder:
    """
    Builder class for creating and configuring components of a Retrieval-Augmented Generation (RAG)
    or Retrieval-Augmented Thinking (RAT) pipeline.

    Attributes:
        vector_store (Optional[VectorStore]): The configured vector store instance.
        embeddings (Optional[EmbeddingsModel]): The configured embeddings model instance.
        llm (Optional[LLM]): The configured large language model (LLM) instance.
        reasoning_llm (Optional[LLM]): The configured reasoning LLM instance for RAT pipelines.
        rag (Optional[RAG]): The configured RAG pipeline instance.
        rat (Optional[RAT]): The configured RAT pipeline instance.
    """

    def __init__(self) -> None:
        """
        Initializes a Builder instance with no configured components.
        """
        self.vector_store: Optional[VectorStore] = None
        self.embeddings: Optional[EmbeddingsModel] = None
        self.llm: Optional[LLM] = None
        self.reasoning_llm: Optional[LLM] = None
        self.rag: Optional[RAG] = None
        self.rat: Optional[RAT] = None

    def with_embeddings(self, type: str, **kwargs) -> Builder:
        """
        Configures the embeddings model.

        Args:
            type (str): The type of embeddings model to create (e.g., HUGGINGFACE).
            **kwargs: Additional parameters required to initialize the embeddings model.

        Returns:
            Builder: The current instance of the Builder for method chaining.

        Raises:
            ValueError: If an unknown embeddings model type is specified.
        """
        logging.info("⏳ Creating an Embeddings Model...")
        if type == Settings.HUGGINGFACE:
            self.embeddings = HuggingfaceEmbeddingsModel(**kwargs)
        else:
            raise ValueError(f"Unknown Embeddings Model type: {type}")
        logging.info("✅ Embeddings Model created")
        return self

    def with_vector_store(self, type: str, **kwargs) -> Builder:
        """
        Configures the vector store.

        Args:
            type (str): The type of vector store to create (e.g., CHROMA).
            **kwargs: Additional parameters required to initialize the vector store.

        Returns:
            Builder: The current instance of the Builder for method chaining.

        Raises:
            ValueError: If the embeddings model is not set or an unknown vector store type is specified.
        """
        logging.info("⏳ Creating a VectorStore...")
        if self.embeddings is None:
            raise ValueError(
                "You need to set an embedding model before setting a vector store"
            )
        elif type == Settings.CHROMA:
            self.vector_store = ChromaVS(embeddings_model=self.embeddings, **kwargs)
        else:
            raise ValueError(f"Unknown VectorStore type: {type}")
        logging.info("✅ VectorStore created")
        return self

    def with_llm(self, type: str, **kwargs) -> Builder:
        """
        Configures the large language model (LLM).

        Args:
            type (str): The type of LLM to create (e.g., OLLAMA).
            **kwargs: Additional parameters required to initialize the LLM.

        Returns:
            Builder: The current instance of the Builder for method chaining.

        Raises:
            ValueError: If an unknown LLM type is specified.
        """
        logging.info("⏳ Creating an LLM...")
        if type == Settings.OLLAMA:
            self.llm = OllamaModel(**kwargs)
        elif type == Settings.LMSTUDIO:
            self.llm = LMStudioModel(**kwargs)
        else:
            raise ValueError(f"Unknown LLM type: {type}")
        logging.info("✅ LLM created")
        return self

    def with_reasoning_llm(self, type: str, **kwargs) -> Builder:
        """
        Configures the reasoning large language model (LLM) for RAT pipelines.

        Args:
            type (str): The type of LLM to create (e.g., deepseek-r1).
            **kwargs: Additional parameters required to initialize the LLM.

        Returns:
            Builder: The current instance of the Builder for method chaining.

        Raises:
            ValueError: If an unknown or invalid LLM type is specified.
        """
        logging.info("⏳ Creating a reasoning LLM...")
        model_name = kwargs.get("model_name", "")
        if not any(model_name.startswith(prefix) for prefix in Settings.REASONING_LLMS):
            raise ValueError(
                f"Invalid LLM type: {type}. Must start with one of {Settings.REASONING_LLMS}."
            )

        if type == Settings.OLLAMA:
            self.reasoning_llm = OllamaModel(**kwargs)
        elif type == Settings.LMSTUDIO:
            self.reasoning_llm = LMStudioModel(**kwargs)
        else:
            raise ValueError(f"Unknown LLM type: {type}")

        logging.info("✅ Reasoning LLM created")
        return self

    def build_rag(self) -> RAG:
        """
        Builds the RAG pipeline with the configured components.

        Returns:
            RAG: The fully configured RAG pipeline instance.

        Raises:
            ValueError: If any of the required components (vector store, LLM, embeddings) are not set.
        """
        if self.vector_store is None:
            raise ValueError("VectorStore is required")
        if self.llm is None:
            raise ValueError("LLM is required")
        if self.embeddings is None:
            raise ValueError("Embeddings Model is required")
        logging.info("⏳ Building the RAG pipeline...")
        self.rag = RAG(self.embeddings, self.vector_store, self.llm)
        logging.info("✅ RAG pipeline created")
        return self.rag

    def build_rat(self, reflection: int = 1) -> RAT:
        """
        Builds the RAT pipeline with the configured components.

        Args:
            reflection (int, optional): The number of reasoning iterations to perform. Defaults to 1.

        Returns:
            RAT: The fully configured RAT pipeline instance.

        Raises:
            ValueError: If any of the required components (vector store, LLM, reasoning LLM, embeddings) are not set.
        """
        if self.vector_store is None:
            raise ValueError("VectorStore is required")
        if self.llm is None:
            raise ValueError("LLM is required")
        if self.reasoning_llm is None:
            raise ValueError("Reasoning LLM is required")
        if self.embeddings is None:
            raise ValueError("Embeddings Model is required")
        logging.info("⏳ Building the RAT pipeline...")
        self.rat = RAT(
            self.embeddings, self.vector_store, self.reasoning_llm, self.llm, reflection
        )
        logging.info("✅ RAT pipeline created")
        return self.rat

    def build_vector_store(self) -> VectorStore:
        """
        Returns the configured vector store instance.

        Returns:
            VectorStore: The configured vector store instance.

        Raises:
            ValueError: If the vector store or embeddings model is not set.
        """
        if self.vector_store is None:
            raise ValueError("VectorStore is required")
        if self.embeddings is None:
            raise ValueError("Embeddings Model is required")
        logging.info("✅ VectorStore instance returned")
        return self.vector_store

    def build_llm(self) -> LLM:
        """
        Returns the configured LLM instance.

        Returns:
            LLM: The configured large language model instance.

        Raises:
            ValueError: If the LLM is not set.
        """
        if self.llm is None:
            raise ValueError("LLM is required")
        logging.info("✅ LLM instance returned")
        return self.llm
