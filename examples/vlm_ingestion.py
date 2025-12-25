from raglight.document_processing.vlm_pdf_processor import VlmPDFProcessor
from raglight.llm.mistral_model import MistralModel
from raglight.rag.builder import Builder
from raglight.config.settings import Settings

from dotenv import load_dotenv

load_dotenv()
Settings.setup_logging()

persist_directory = "./defaultDb"
model_embeddings = "nomic-embed-text:137m-v1.5-fp16"
collection_name = Settings.DEFAULT_COLLECTION_NAME
data_path = ""  # Path to your data
model_name = "mistral-large-2512"

vlm = MistralModel(
    model_name=model_name,
    system_prompt="You are a technical documentation visual assistant.",
)

custom_processors = {
    "pdf": VlmPDFProcessor(vlm),  # override default PDFProcessor
}

rag = (
    Builder()
    .with_embeddings(Settings.OLLAMA, model_name=model_embeddings)
    .with_vector_store(
        Settings.CHROMA,
        persist_directory=persist_directory,
        collection_name=collection_name,
        custom_processors=custom_processors,
    )
    .with_llm(
        Settings.MISTRAL,
        model_name=model_name,
        system_prompt="Please respond to user answer",
    )
    .build_rag(k=15)
)

rag.vector_store.ingest(data_path=data_path)

response = rag.generate("Please explain PID functionment")
print(response)
