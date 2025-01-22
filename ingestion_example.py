from src.raglight.rag.builder import Builder
from src.raglight.config.settings import Settings
from dotenv import load_dotenv
import os

load_dotenv()
Settings.setup_logging()

persist_directory = os.environ.get('PERSIST_DIRECTORY_INGESTION')
model_embeddings = os.environ.get('MODEL_EMBEDDINGS')
collection_name = os.environ.get('COLLECTION_NAME_INGESTION')
data_path = os.environ.get('DATA_PATH')

file_extension ='**/*.pdf'

vector_store = Builder() \
.with_embeddings(Settings.HUGGINGFACE, model_name=model_embeddings) \
.with_vector_store(Settings.CHROMA, persist_directory=persist_directory, collection_name=collection_name) \
.build_vector_store()

vector_store.ingest(file_extension=file_extension, data_path=data_path)
