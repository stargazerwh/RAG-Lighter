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
