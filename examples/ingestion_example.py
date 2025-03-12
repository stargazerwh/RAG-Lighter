from raglight.rag.builder import Builder
from raglight.config.settings import Settings
from dotenv import load_dotenv
import os

load_dotenv()
Settings.setup_logging()

persist_directory = './defaultDb'
model_embeddings = Settings.DEFAULT_EMBEDDINGS_MODEL
collection_name = Settings.DEFAULT_COLLECTION_NAME
data_path = os.environ.get('DATA_PATH')

file_extension = Settings.DEFAULT_EXTENSIONS # All visible files

vector_store = Builder() \
.with_embeddings(Settings.HUGGINGFACE, model_name=model_embeddings) \
.with_vector_store(Settings.CHROMA, persist_directory=persist_directory, collection_name=collection_name) \
.build_vector_store()

vector_store.ingest(file_extension=file_extension, data_path=data_path)
