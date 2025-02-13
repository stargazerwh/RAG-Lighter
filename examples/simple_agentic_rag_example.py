from src.raglight.rag.simple_agentic_rag_api import AgenticRAGPipeline
from src.raglight.models.data_source_model import FolderSource, GitHubSource
from src.raglight.config.settings import Settings

Settings.setup_logging()

pipeline = AgenticRAGPipeline(knowledge_base=[
    # FolderSource(path="<path to your folder with pdf>/knowledge_base"),
    GitHubSource(url="https://github.com/Bessouat40/RAGLight")
    ],
    model_name="llama3",
    provider=Settings.OLLAMA,
    k=5)

pipeline.build()

response = pipeline.generate("How can I create an easy RAGPipeline using raglight framework ? Give me python implementation")
print(response)
