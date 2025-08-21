from smolagents import Tool, tool, MCPClient
from smolagents import CodeAgent, OpenAIServerModel, LiteLLMModel

from ..config.vector_store_config import VectorStoreConfig
from ..config.settings import Settings
from ..config.agentic_rag_config import AgenticRAGConfig
from ..vectorstore.vector_store import VectorStore
from ..rag.builder import Builder


class RetrieverTool(Tool):
    name = "retriever"
    description = (
        "Uses semantic search to retrieve relevant parts of the code documentation."
    )

    inputs = {
        "query": {
            "type": "string",
            "description": "The query to perform. Should be semantically close to the target documents.",
        },
    }
    output_type = "string"

    def __init__(self, k: int, vector_store: VectorStore, **kwargs):
        super().__init__(**kwargs)
        self.vector_store: VectorStore = vector_store
        self.k: int = k

    def forward(self, query: str) -> str:

        retrieved_docs = self.vector_store.similarity_search(query, k=self.k)

        return "\nRetrieved documents:\n" + "".join(
            [
                f"\n\n===== Document {str(i)} =====\n" + doc.page_content
                for i, doc in enumerate(retrieved_docs)
            ]
        )


class ClassRetrieverTool(Tool):
    """
    Retrieves class definitions from the codebase.
    """

    name = "class_retriever"
    description = "Retrieves class definitions and their locations in the codebase."

    inputs = {
        "query": {
            "type": "string",
            "description": "The name or description of the class to retrieve.",
        },
    }
    output_type = "string"

    def __init__(self, k: int, vector_store: VectorStore, **kwargs):
        super().__init__(**kwargs)
        self.vector_store: VectorStore = vector_store
        self.k: int = k

    def forward(self, query: str) -> str:

        retrieved_classes = self.vector_store.similarity_search_class(query, k=self.k)

        return "\nRetrieved classes:\n" + "".join(
            [
                f"\n\n===== Class {str(i)} =====\n"
                + doc.page_content
                + f"\nSource File: {doc.metadata['source']}"
                for i, doc in enumerate(retrieved_classes)
            ]
        )


class AgenticRAG:
    def __init__(
        self,
        config: AgenticRAGConfig,
        vector_store_config: VectorStoreConfig,
    ):
        self.config = config
        self.mcp_configs = config.mcp_config or []
        self.vector_store = self.create_vector_store(vector_store_config)

        self.k: int = config.k

        self.local_tools = [
            RetrieverTool(k=config.k, vector_store=self.vector_store),
            ClassRetrieverTool(k=config.k, vector_store=self.vector_store),
        ]

        self.model = self._create_llm_model(config)

    def _create_llm_model(self, config: AgenticRAGConfig):
        """Crée l'instance du modèle LLM à partir de la config."""
        if config.provider.lower() == Settings.MISTRAL.lower():
            api_base = config.api_base or Settings.MISTRAL_API
            return OpenAIServerModel(
                model_id=config.model,
                api_key=Settings.MISTRAL_API_KEY,
                api_base=api_base,
            )
        elif config.provider == Settings.OPENAI:
            api_base = config.api_base or Settings.DEFAULT_OPENAI_CLIENT
            return OpenAIServerModel(
                model_id=config.model,
                api_key=Settings.OPENAI_API_KEY,
                api_base=api_base,
            )
        elif config.provider == Settings.GOOGLE_GEMINI:
            api_base = config.api_base or Settings.DEFAULT_GOOGLE_CLIENT
            return OpenAIServerModel(
                model_id=config.model,
                api_base=api_base,
                api_key=config.api_key or Settings.GEMINI_API_KEY,
            )
        else:
            api_base = config.api_base or Settings.DEFAULT_OLLAMA_CLIENT
            return LiteLLMModel(
                model_id=f"{config.provider.lower()}/{config.model}",
                api_base=config.api_base,
                api_key=config.api_key,
                num_ctx=config.num_ctx,
            )

    def generate(
        self, query: str, search_type: str = "code", stream: bool = False
    ) -> str:
        """
        Generates a response using the appropriate retrieval tool.

        Args:
            query (str): The search query.
            metadata_filter (dict, optional): Metadata filter (e.g., {'source': 'file.py'} or {'classes': 'MyClass'}).
            search_type (str): Either "code" for full document retrieval or "class" for class retrieval.

        Returns:
            str: The retrieved information.
        """
        task_instruction = f"Query: {query}"
        task_instruction += (
            f"\nTool: {'class_retriever' if search_type == 'class' else 'retriever'}"
        )
        if not self.mcp_configs:
            agent = CodeAgent(
                tools=self.local_tools,
                model=self.model,
                max_steps=self.config.max_steps,
                verbosity_level=self.config.verbosity_level,
            )
            return agent.run(task_instruction, stream)
        else:
            with MCPClient(self.mcp_configs) as mcp_tools:
                agent = CodeAgent(
                    tools=[*self.local_tools, *mcp_tools],
                    model=self.model,
                    max_steps=self.config.max_steps,
                    verbosity_level=self.config.verbosity_level,
                )
                return agent.run(task_instruction, stream)

    def create_vector_store(self, config: VectorStoreConfig) -> VectorStore:
        """Creates a vector store using the provided configuration.

        Args:
            config (VectorStoreConfig): The configuration for the vector store.

        Returns:
            VectorStore: An instance of the vector store.
        """
        return (
            Builder()
            .with_embeddings(
                config.provider,
                model_name=config.embedding_model,
                api_base=config.api_base,
            )
            .with_vector_store(
                type=config.database,
                persist_directory=config.persist_directory,
                collection_name=config.collection_name,
                host=config.host,
                port=config.port,
            )
            .build_vector_store()
        )
