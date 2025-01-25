from raglight.rat.simple_rat_api import RATPipeline
from raglight.models.data_source_model import FolderSource, GitHubSource
from raglight.config.settings import Settings

Settings.setup_logging()

pipeline = RATPipeline(knowledge_base=[
# FolderSource(path="<path to your folder with pdf>/knowledge_base"),
GitHubSource(url="https://github.com/Bessouat40/RAGLight"),
GitHubSource(url="https://github.com/Bessouat40/LLMChat")
], model_name="llama3", reasoning_model_name="deepseek-r1:1.5b")

pipeline.build()

response = pipeline.generate("How can I create a RAG using the class Builder of RAGLight framework ?")
print(response)

