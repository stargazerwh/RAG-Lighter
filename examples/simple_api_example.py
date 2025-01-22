from raglight.rag.simple_rag_api import RAGPipeline
from raglight.config.settings import Settings

Settings.setup_logging()

pipeline = RAGPipeline(knowledge_base=[
{"name": "PDFs", "type": "folder", "path": "<path to your folder with pdf>/grece"},
])

pipeline.build()

response = pipeline.generate("Tell me something about grece mythology please")
print(response)