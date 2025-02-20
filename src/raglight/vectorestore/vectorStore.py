from abc import ABC, abstractmethod
from typing import Any, List
import os
import re
import ast
import logging
from langchain_text_splitters import Language
from ..embeddings.embeddingsModel import EmbeddingsModel


class VectorStore(ABC):
    """
    Abstract base class for vector store implementations.

    This class provides a blueprint for creating vector stores that handle document ingestion,
    indexing, and similarity search. Concrete implementations must define the methods for
    specific vector store backends.

    Attributes:
        persist_directory (str): The directory where the vector store data is persisted.
        embeddings_model: The embeddings model used for encoding documents.
        vector_store: The actual instance of the vector store, initialized in the subclass.
    """

    def __init__(
        self, persist_directory: str, embeddings_model: EmbeddingsModel
    ) -> None:
        """
        Initializes a VectorStore instance.

        Args:
            persist_directory (str): Directory where the vector store data will be persisted.
            embeddings_model (Any): The embeddings model instance used for vectorization.
        """
        self.JAVA_TS_CPP_CLASS_PATTERN = r"\bclass\s+(\w+)"
        self.CSHARP_CLASS_PATTERN = r"\bclass\s+(\w+)"
        self.embeddings_model: Any = embeddings_model.get_model()
        self.persist_directory: str = persist_directory
        self.vector_store: Any = None

    @abstractmethod
    def ingest(self, **kwargs: Any) -> None:
        """
        Abstract method to ingest and index documents in the vector store.

        Args:
            **kwargs (Any): Additional parameters required for ingestion, depending on the implementation.
        """
        pass

    @abstractmethod
    def ingest_code(self, **kwargs: Any) -> None:
        """
        Abstract method to ingest and index code in the vector store.

        Args:
            **kwargs (Any): Additional parameters required for ingestion, depending on the implementation.
        """
        pass

    @abstractmethod
    def similarity_search(self, question: str, k: int = 2) -> List[Any]:
        """
        Abstract method to perform similarity search in the vector store.

        Args:
            question (str): The input query for similarity search.
            k (int, optional): The number of top results to retrieve. Defaults to 2.

        Returns:
            List[Any]: A list of top-k similar documents.
        """
        pass

    @abstractmethod
    def similarity_search_class(self, question: str, k: int = 2) -> List[Any]:
        """
        Abstract method to perform similarity search in the vector store.

        Args:
            question (str): The input query for similarity search.
            k (int, optional): The number of top results to retrieve. Defaults to 2.

        Returns:
            List[Any]: A list of top-k similar documents.
        """
        pass

    def get_language_from_extension(self, extension: str) -> Language | None:
        """
        Maps a file extension to a Language enum.

        Args:
            extension (str): File extension (e.g., 'py', 'js').

        Returns:
            Language: Corresponding Language enum, or None if not supported.
        """
        extension_to_language = {
            "py": Language.PYTHON,
            "js": Language.JS,
            "ts": Language.TS,
            "java": Language.JAVA,
            "cpp": Language.CPP,
            "go": Language.GO,
            "php": Language.PHP,
            "rb": Language.RUBY,
            "rs": Language.RUST,
            "scala": Language.SCALA,
            "swift": Language.SWIFT,
            "md": Language.MARKDOWN,
            "html": Language.HTML,
            "sol": Language.SOL,
            "cs": Language.CSHARP,
            "c": Language.C,
            "lua": Language.LUA,
            "pl": Language.PERL,
            "hs": Language.HASKELL,
        }
        return extension_to_language.get(extension)

    def get_classes(self, repos_path: str) -> dict:
        """
        Extracts all class names and their signatures from the project's source code.

        Args:
            repos_path (str): Path to the project's root directory.

        Returns:
            dict: A dictionary where keys are file paths and values are lists of class signatures.
        """
        class_map = {}

        for root, _, files in os.walk(repos_path):
            for file in files:
                file_path = os.path.join(root, file)
                file_extension = os.path.splitext(file)[1][1:]
                language = self.get_language_from_extension(file_extension)
                class_signatures = []

                if language:
                    try:
                        logging.info(f"🔍 Extracting classes from {file_path}")

                        if language == Language.PYTHON:
                            class_signatures = self.extract_python_class_signatures(
                                file_path
                            )
                        elif language in {
                            Language.JS,
                            Language.TS,
                            Language.JAVA,
                            Language.CPP,
                            Language.CSHARP,
                        }:
                            class_signatures = self.extract_class_signatures_with_regex(
                                file_path, language
                            )

                        if class_signatures:
                            class_map[file_path] = class_signatures

                    except Exception as e:
                        logging.warning(
                            f"⚠️ Error extracting classes from {file_path}: {e}"
                        )

        logging.info(
            f"✅ Extracted {sum(len(v) for v in class_map.values())} class signatures"
        )
        return class_map

    def extract_python_class_signatures(self, file_path: str) -> List[str]:
        """
        Extracts class signatures (name + inheritance) from a Python file.

        Args:
            file_path (str): Path to the Python file.

        Returns:
            List[str]: List of class signatures.
        """
        with open(file_path, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=file_path)

        class_signatures = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                bases = [
                    base.id if isinstance(base, ast.Name) else "?"
                    for base in node.bases
                ]
                class_signature = f"class {node.name}({', '.join(bases)})"
                class_signatures.append(class_signature)

        return class_signatures

    def extract_class_signatures_with_regex(
        self, file_path: str, language: Language
    ) -> List[str]:
        """
        Extracts class signatures using regex for various languages.

        Args:
            file_path (str): Path to the source code file.
            language (Language): The programming language.

        Returns:
            List[str]: List of class signatures.
        """
        patterns = {
            Language.JAVA: r"class\s+(\w+)\s*(?:extends\s+(\w+))?\s*(?:implements\s+([\w, ]+))?",
            Language.JS: r"class\s+(\w+)\s*(?:extends\s+(\w+))?",
            Language.TS: r"class\s+(\w+)\s*(?:extends\s+(\w+))?",
            Language.CPP: r"class\s+(\w+)\s*(?::\s*(public|private|protected)?\s*(\w+))?",
            Language.CSHARP: r"class\s+(\w+)\s*(?::\s*(\w+))?",
        }

        pattern = patterns.get(language)
        if not pattern:
            return []

        with open(file_path, "r", encoding="utf-8") as f:
            code = f.read()

        matches = re.findall(pattern, code)
        return [f"class {m[0]}({', '.join(filter(None, m[1:]))})" for m in matches]
