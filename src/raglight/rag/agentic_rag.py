from typing import TypedDict, Dict
from smolagents import Tool, tool
from smolagents import CodeAgent, OpenAIServerModel, LiteLLMModel

from ..config.vector_store_config import VectorStoreConfig
from ..config.settings import Settings
from ..config.agentic_rag_config import AgenticRAGConfig
from ..vectorestore.vectorStore import VectorStore
from ..rag.builder import Builder

import json


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
        self, config: AgenticRAGConfig, vector_store_config: VectorStoreConfig
    ):
        self.vector_store = self.create_vector_store(vector_store_config)

        self.k: int = config.k

        retriever_tool = RetrieverTool(k=config.k, vector_store=self.vector_store)
        class_retriever_tool = ClassRetrieverTool(
            k=config.k, vector_store=self.vector_store
        )

        if config.provider.lower() == Settings.MISTRAL.lower():
            model = OpenAIServerModel(
                model_id=config.model,
                api_key=Settings.MISTRAL_API_KEY,
                api_base=Settings.MISTRAL_API,
            )

        else:
            model = LiteLLMModel(
                model_id=f"{config.provider.lower()}/{config.model}",
                api_base=config.api_base,
                api_key=config.api_key,
                num_ctx=config.num_ctx,
            )
        self.agent = CodeAgent(
            tools=[retriever_tool, class_retriever_tool],
            model=model,
            max_steps=config.max_steps,
            verbosity_level=config.verbosity_level,
            prompt_templates=create_prompt_templates(config.system_prompt),
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

        return self.agent.run(task_instruction, stream)

    def create_vector_store(self, config: VectorStoreConfig) -> VectorStore:
        """Creates a vector store using the provided configuration.

        Args:
            config (VectorStoreConfig): The configuration for the vector store.

        Returns:
            VectorStore: An instance of the vector store.
        """
        return (
            Builder()
            .with_embeddings(config.provider, model_name=config.embedding_model)
            .with_vector_store(
                type=config.database,
                persist_directory=config.persist_directory,
                collection_name=config.collection_name,
            )
            .build_vector_store()
        )


class PlanningPromptTemplate(TypedDict):
    """
    Prompt templates for the planning step.
    """

    initial_facts_pre_task: str
    initial_facts_task: str
    initial_plan: str
    update_facts_pre_messages: str
    update_facts_post_messages: str
    update_plan_pre_messages: str
    update_plan_post_messages: str


class ManagedAgentPromptTemplate(TypedDict):
    """
    Prompt templates for the managed agent.
    """

    task: str
    report: str


class FinalAnswerPromptTemplate(TypedDict):
    """
    Prompt templates for the final answer.
    """

    pre_messages: str
    post_messages: str


class PromptTemplates(TypedDict):
    """
    Prompt templates for the agent.
    """

    system_prompt: str
    planning: PlanningPromptTemplate
    managed_agent: ManagedAgentPromptTemplate
    final_answer: FinalAnswerPromptTemplate


def create_prompt_templates(
    system_prompt: str = Settings.DEFAULT_AGENT_PROMPT,
) -> PromptTemplates:
    return PromptTemplates(
        system_prompt=system_prompt,
        planning=PlanningPromptTemplate(
            initial_facts="",
            initial_plan="",
            update_facts_pre_messages="",
            update_facts_post_messages="",
            update_plan_pre_messages="",
            update_plan_post_messages="",
        ),
        managed_agent=ManagedAgentPromptTemplate(task="", report=""),
        final_answer=FinalAnswerPromptTemplate(pre_messages="", post_messages=""),
    )
