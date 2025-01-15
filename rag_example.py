from raglight.rag.builder import Builder
from raglight.config.settings import Settings
from dotenv import load_dotenv
import os

load_dotenv()

persist_directory = os.environ.get('PERSIST_DIRECTORY')
model_embeddings = os.environ.get('MODEL_EMBEDDINGS')
model_name = os.environ.get('MODEL_NAME')
system_prompt_directory = os.environ.get('SYSTEM_PROMPT_DIRECTORY')
collection_name = os.environ.get('COLLECTION_NAME')

rag = Builder() \
.with_embeddings(Settings.HUGGINGFACE, model_name=model_embeddings) \
.with_vector_store(Settings.CHROMA, persist_directory=persist_directory, collection_name=collection_name) \
.with_llm(Settings.OLLAMA, model_name=model_name, system_prompt_file=system_prompt_directory) \
.build_rag()

rag.question_graph("How to program a marathon preparation ?")
