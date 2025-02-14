from raglight.rag.simple_agentic_rag_api import AgenticRAGPipeline
from raglight.config.agentic_rag_config import SimpleAgenticRAGConfig
from raglight.models.data_source_model import FolderSource, GitHubSource
from raglight.config.settings import Settings

Settings.setup_logging()

config = config = SimpleAgenticRAGConfig(
    #   provider = Settings.OLLAMA.lower(), # default "ollama"
    #   model = Settings.DEFAULT_LLM, # default "llama3"
    #   k= 5,
    #   max_steps = 4,
    #   system_prompt = Settings.DEFAULT_AGENT_PROMPT
    #   api_key="YOUR_API_KEY",
    #   api_base: Settings.DEFAULT_OLLAMA_CLIENT
    #   num_ctx: 8192
    #   verbosity_level: 2
    )

pipeline = AgenticRAGPipeline(knowledge_base=[
    FolderSource(path="<path to your folder with pdf>/knowledge_base"),
    GitHubSource(url="https://github.com/Bessouat40/RAGLight")
    ],
    config=config)

pipeline.build()

response = pipeline.generate("How can I create an easy RAGPipeline using raglight framework ? Give me python implementation")
print(response)
