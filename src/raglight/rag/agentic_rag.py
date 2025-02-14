from typing import TypedDict
from smolagents import Tool, tool
from smolagents import CodeAgent
from smolagents import LiteLLMModel

from ..config.settings import Settings
from ..config.agentic_rag_config import AgenticRAGConfig
from ..vectorestore.vectorStore import VectorStore


class RetrieverTool(Tool):
    name = "retriever"
    description = "Uses semantic search to retrieve the parts of transformers documentation that could be most relevant to answer your query."
    inputs = {
        "query": {
            "type": "string",
            "description": "The query to perform. This should be semantically close to your target documents. Use the affirmative form rather than a question.",
        }
    }
    output_type = "string"

    def __init__(self, config: AgenticRAGConfig, **kwargs):
        super().__init__(**kwargs)
        self.vector_store: VectorStore = config.vector_store
        self.k: int = config.k

    def forward(self, query: str) -> str:
        retrieved_docs = self.vector_store.similarity_search(query, k=self.k)
        return "\nRetrieved documents:\n" + "".join(
            [
                f"\n\n===== Document {str(i)} =====\n" + doc.page_content
                for i, doc in enumerate(retrieved_docs)
            ]
        )


class AgenticRAG:

    def __init__(self, config: AgenticRAGConfig):
        self.vector_store: VectorStore = config.vector_store
        self.k: int = config.k
        retriever_tool = RetrieverTool(config=config)
        self.agent = CodeAgent(
            tools=[retriever_tool],
            model=LiteLLMModel(
                model_id=f"{config.provider}/{config.model}",
                api_base=config.api_base,
                api_key=config.api_key,
                num_ctx=config.num_ctx,
            ),
            max_steps=config.max_steps,
            verbosity_level=config.verbosity_level,
            prompt_templates=PromptTemplates(),
        )

    def generate(self, query: str) -> str:
        return self.agent.run(query)


class PlanningPromptTemplate(TypedDict):
    """
    Prompt templates for the planning step.

    Args:
        initial_facts_pre_task (`str`): Initial facts pre-task prompt.
        initial_facts_task (`str`): Initial facts task prompt.
        initial_plan (`str`): Initial plan prompt.
        update_facts_pre_messages (`str`): Update facts pre-messages prompt.
        update_facts_post_messages (`str`): Update facts post-messages prompt.
        update_plan_pre_messages (`str`): Update plan pre-messages prompt.
        update_plan_post_messages (`str`): Update plan post-messages prompt.
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

    Args:
        task (`str`): Task prompt.
        report (`str`): Report prompt.
    """

    task: str
    report: str


class FinalAnswerPromptTemplate(TypedDict):
    """
    Prompt templates for the final answer.

    Args:
        pre_messages (`str`): Pre-messages prompt.
        post_messages (`str`): Post-messages prompt.
    """

    pre_messages: str
    post_messages: str


class PromptTemplates(TypedDict):
    """
    Prompt templates for the agent.

    Args:
        system_prompt (`str`): System prompt.
        planning ([`~agents.PlanningPromptTemplate`]): Planning prompt templates.
        managed_agent ([`~agents.ManagedAgentPromptTemplate`]): Managed agent prompt templates.
        final_answer ([`~agents.FinalAnswerPromptTemplate`]): Final answer prompt templates.
    """

    system_prompt: str = Settings.DEFAULT_AGENT_PROMPT
    planning = (
        PlanningPromptTemplate(
            initial_facts="",
            initial_plan="",
            update_facts_pre_messages="",
            update_facts_post_messages="",
            update_plan_pre_messages="",
            update_plan_post_messages="",
        ),
    )
    managed_agent = (ManagedAgentPromptTemplate(task="", report=""),)
    final_answer = (FinalAnswerPromptTemplate(pre_messages="", post_messages=""),)
