from raglight.rag.simple_rag_api import RAGPipeline
from raglight.models.data_source_model import FolderSource, GitHubSource
from raglight.config.settings import Settings

Settings.setup_logging()

pipeline = RAGPipeline(knowledge_base=[
    FolderSource(path="<path to your folder with pdf>/knowledge_base"),
    GitHubSource(url="https://github.com/Bessouat40/RAGLight")
    ],
    model_name="llama3",
    provider=Settings.OLLAMA,
    k=5)

pipeline.build()

response = pipeline.generate("How can I create an easy RAGPipeline using raglight framework ? Give me python implementation")
print(response)
