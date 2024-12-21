from src.rag.builder import Builder
from dotenv import load_dotenv
import os

load_dotenv()

persist_directory = os.environ.get('PERSIST_DIRECTORY')
model_embeddings = os.environ.get('MODEL_EMBEDDINGS')
model_name = os.environ.get('MODEL_NAME')
system_prompt_directory = os.environ.get('SYSTEM_PROMPT_DIRECTORY')
collection_name = os.environ.get('COLLECTION_NAME')

rag = Builder() \
.with_embeddings('HuggingFace', model_name=model_embeddings) \
.with_vector_store('Chroma', persist_directory=persist_directory, collection_name=collection_name) \
.with_llm('Ollama', model_name=model_name, system_prompt_file=system_prompt_directory) \
.build_rag()

#RAG retrieve data in french so question is in french
rag.question_graph("Comment gérer mon alimentation pendant un marathon ?")