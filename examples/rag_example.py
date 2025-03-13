from raglight.rag.builder import Builder
from raglight.config.settings import Settings
from dotenv import load_dotenv

load_dotenv()
Settings.setup_logging()

persist_directory = './defaultDb'
model_embeddings = Settings.DEFAULT_EMBEDDINGS_MODEL
model_name = 'llama3.2:3b'
system_prompt_directory = Settings.DEFAULT_SYSTEM_PROMPT
collection_name = Settings.DEFAULT_COLLECTION_NAME

rag = Builder() \
    .with_embeddings(Settings.HUGGINGFACE, model_name=model_embeddings) \
    .with_vector_store(Settings.CHROMA, persist_directory=persist_directory, collection_name=collection_name) \
    .with_llm(Settings.OLLAMA, model_name=model_name, system_prompt=system_prompt_directory) \
    .build_rag(k = 5)

response = rag.generate("Can you tell me something about grece mythology ?")
print(response)

