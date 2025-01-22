import logging


class Settings:
    """
    A class that contains constants for various settings used in the project.
    """

    @staticmethod
    def setup_logging() -> None:
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
        )

    CHROMA = "Chroma"
    OLLAMA = "Ollama"
    HUGGINGFACE = "HuggingFace"
    DEFAULT_LLM = "llama3"
    DEFAULT_EMBEDDINGS_MODEL = "all-MiniLM-L6-v2"
    DEFAULT_SYSTEM_PROMPT = "You are an assistant and you need to response to user query using provided informations."
    DEFAULT_COLLECTION_NAME = "default"
    DEFAULT_PERSIST_DIRECTORY = "./defaultDb"
    DEFAULT_OLLAMA_CLIENT = "http://localhost:11434"
    DEFAULT_EXTENSIONS = "**/[!.]*"
