from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain_mistralai import ChatMistralAI
from langchain_ollama import ChatOllama
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_mcp_adapters.client import MultiServerMCPClient  
from langchain.messages import SystemMessage, HumanMessage

from .agentic_rag_utils.rag_tools import RetrieverTool, ClassRetrieverTool
from .builder import Builder
from ..vectorstore.vector_store import VectorStore
from ..config.agentic_rag_config import AgenticRAGConfig
from ..config.settings import Settings
from ..config.vector_store_config import VectorStoreConfig

class AgenticRAG:
    """
    Main engine for the Agentic RAG pipeline.

    This class orchestrates the interaction between the Large Language Model (LLM),
    local Vector Store tools, and optional Model Context Protocol (MCP) servers
    to generate grounded responses.
    """

    def __init__(
        self,
        config: AgenticRAGConfig,
        vector_store_config: VectorStoreConfig,
    ):
        """
        Initializes the Agentic RAG engine.

        Args:
            config (AgenticRAGConfig): Configuration for the RAG agent (LLM, MCP, etc.).
            vector_store_config (VectorStoreConfig): Configuration for the underlying vector database.
        """
        self.config = config
        self.mcp_configs = config.mcp_config or []
        self.vector_store = self.create_vector_store(vector_store_config)

        self.k: int = config.k

        self.local_tools = [
            RetrieverTool(k=config.k, vector_store=self.vector_store),
            ClassRetrieverTool(k=config.k, vector_store=self.vector_store),
        ]

        try:
            if hasattr(self.vector_store, "get_available_collections"):
                collections = self.vector_store.get_available_collections()
                if collections:
                    cols_list = ", ".join([str(c) for c in collections])
                    for tool in self.local_tools:
                        tool.description += f" Available collections in the database: {cols_list}."
        except Exception:
            # Fail silently if vector store does not implement the method properly yet
            pass

        self.model = self._create_llm_model(config)

    def _create_llm_model(self, config: AgenticRAGConfig) -> BaseChatModel:
        """
        Factory method that instantiates the appropriate LangChain ChatModel.

        Based on the provider specified in the configuration (e.g., OpenAI, Mistral, Ollama),
        this method returns a configured ChatModel instance ready for the agent.

        Args:
            config (AgenticRAGConfig): The configuration object containing provider settings.

        Returns:
            BaseChatModel: A LangChain-compatible chat model instance.

        Raises:
            ValueError: If the specified provider is not supported.
        """
        provider = config.provider
        
        if provider == Settings.GOOGLE_GEMINI:
            return ChatGoogleGenerativeAI(
                model=config.model,
                google_api_key=config.api_key or Settings.GEMINI_API_KEY,
                temperature=0.5,
            )

        elif provider == Settings.MISTRAL:
            return ChatMistralAI(
                model=config.model,
                mistral_api_key=config.api_key or Settings.MISTRAL_API_KEY,
                temperature=0.5,
            )

        elif provider == Settings.OPENAI:
            return ChatOpenAI(
                model=config.model,
                base_url=config.api_base or Settings.DEFAULT_OPENAI_CLIENT,
                api_key=config.api_key,
                temperature=0.5,
            )

        elif provider == Settings.OLLAMA:
            return ChatOllama(
                model=config.model,
                base_url=config.api_base or Settings.DEFAULT_OLLAMA_CLIENT,
                temperature=0.5,
            )

        elif provider == Settings.LMSTUDIO:
            return ChatOpenAI(
                model=config.model,
                openai_api_key="not-needed",
                base_url=config.api_base or Settings.DEFAULT_LMSTUDIO_CLIENT,
                temperature=0.5,
            )

        else:
            raise ValueError(f"Provider '{provider}' not supported by LangChain factory.")
        
    def create_vector_store(self, config: VectorStoreConfig) -> VectorStore:
        """
        Creates and configures the vector store instance.

        Args:
            config (VectorStoreConfig): The configuration for the vector store.

        Returns:
            VectorStore: An initialized instance of the vector store.
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
    
    async def generate(self, query: str, stream: bool = False) -> str:
        """
        Asynchronously generates a response to the user's query.

        This method:
        1. Detects and configures any Model Context Protocol (MCP) servers defined in the config.
        2. Aggregates local tools (VectorStore retrievers) and remote MCP tools.
        3. Invokes the LangChain agent to process the query.

        Args:
            query (str): The user's question or command.
            stream (bool): Whether to stream the response (currently unused, kept for API compatibility).

        Returns:
            str: The final text response generated by the agent.
        """
        
        mcp_servers_dict = {}
        if self.config.mcp_config:
            config_list = self.config.mcp_config if isinstance(self.config.mcp_config, list) else [self.config.mcp_config]

            for idx, conf in enumerate(config_list):
                server_name = f"server_{idx}"
                
                cmd = getattr(conf, "command", None) or (conf.get("command") if isinstance(conf, dict) else None)
                args = getattr(conf, "args", []) or (conf.get("args", []) if isinstance(conf, dict) else [])
                env = getattr(conf, "env", None) or (conf.get("env") if isinstance(conf, dict) else None)
                url = getattr(conf, "url", None) or (conf.get("url") if isinstance(conf, dict) else None)

                if url:
                    mcp_servers_dict[server_name] = {"transport": "sse", "url": url}
                elif cmd:
                    mcp_servers_dict[server_name] = {"transport": "stdio", "command": cmd, "args": args, "env": env}

        if mcp_servers_dict and MultiServerMCPClient:
            async with MultiServerMCPClient(mcp_servers_dict) as mcp_client:
                mcp_tools = await mcp_client.get_tools()
                all_tools = self.local_tools + mcp_tools
                return await self._run_agent_execution(query, all_tools)
        else:
            return await self._run_agent_execution(query, self.local_tools)

    async def _run_agent_execution(self, query: str, tools: list) -> str:
        """
        Executes the LangChain agent with a specific set of tools.

        This internal helper initializes the agent with the provided tools and the system prompt,
        then sends the user query for processing.

        Args:
            query (str): The user input.
            tools (list): A list of LangChain tools (local and/or MCP) available to the agent.

        Returns:
            str: The agent's output text.
        """
        agent = create_agent(self.model, tools=tools)

        messages = []
        sys_prompt = getattr(self.config, "system_prompt", "You are a helpful coding assistant.")
        messages.append({"role": "system", "content": sys_prompt})
        messages.append({"role": "user", "content": query})

        try:
            response = await agent.ainvoke({"messages": messages})
            
            if isinstance(response, dict) and "messages" in response:
                last_msg = response["messages"][-1]
                return last_msg.content if hasattr(last_msg, "content") else str(last_msg)
            else:
                return response.get("output", str(response))
        except Exception as e:
            raise SystemError(f"Error during generation: {str(e)}")