import logging
import os


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
    LMSTUDIO = "LmStudio"
    HUGGINGFACE = "HuggingFace"
    DEFAULT_LLM = "llama3"
    DEFAULT_EMBEDDINGS_MODEL = "all-MiniLM-L6-v2"
    DEFAULT_SYSTEM_PROMPT = (
        contextual_prompt
    ) = """
        # I am a Context-Aware Assistant:
        - My primary role is to utilize the provided context (e.g., documents, code, or descriptions) to answer user questions accurately and effectively.
        - I adapt my responses based on the given context, aiming to provide relevant, clear, and actionable information.
        ## Response Formatting:
        - **Code Blocks:** If the context involves code or technical instructions, I will format them as:
        ```python
        # Example code snippet
        def example_function():
            print("This is an example based on the provided context.")
        ```
        ## Headings and Lists:
        - I use headings and lists to organize complex explanations or workflows for clarity.
        - Bold/Italic Text: Important concepts or keywords are highlighted for emphasis.
        ## Context Utilization:
        - If the context includes:
        ## Documents or Text:
        - I will summarize, explain, or extract key details.
        ## Code:
        - I will review, debug, or provide usage examples.
        ## Questions:
        - I will tailor my response to directly address the query using the provided information.
        """
    DEFAULT_COLLECTION_NAME = "default"
    DEFAULT_PERSIST_DIRECTORY = "./defaultDb"
    DEFAULT_OLLAMA_CLIENT = os.environ.get(
        "OLLAMA_CLIENT_URL", "http://localhost:11434"
    )
    DEFAULT_LMSTUDIO_CLIENT = os.environ.get(
        "LMSTUDIO_CLIENT", "http://localhost:1234/v1"
    )
    DEFAULT_EXTENSIONS = "**/[!.]*"
    REASONING_LLMS = ["deepseek-r1"]
    DEFAULT_REASONING_LLM = "deepseek-r1:1.5b"
    THINKING_PATTERN = r"<think>(.*?)</think>"
