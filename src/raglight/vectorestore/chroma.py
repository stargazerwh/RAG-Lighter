import glob
import logging
from typing import Any, List, Dict
from typing_extensions import override
from .vectorStore import VectorStore
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
import os
from ..embeddings.embeddingsModel import EmbeddingsModel


class ChromaVS(VectorStore):
    """
    Concrete implementation of the VectorStore class using Chroma as the vector storage backend.

    This class handles the ingestion, indexing, and similarity search operations for document-based
    vector stores with Chroma as the underlying storage.

    Attributes:
        persist_directory (str): The directory where the Chroma index is persisted.
        vector_store (Chroma): The Chroma vector store instance.
        embeddings_model: The embeddings model used to encode the documents.
    """

    def __init__(
        self,
        persist_directory: str,
        collection_name: str,
        embeddings_model: EmbeddingsModel,
    ) -> None:
        """
        Initializes a ChromaVS instance.

        Args:
            persist_directory (str): Directory to persist the Chroma index.
            collection_name (str): Name of the collection within the Chroma vector store.
            embeddings_model (Any): The embeddings model used for vectorization.
        """
        super().__init__(persist_directory, embeddings_model)
        self.vector_store = Chroma(
            embedding_function=self.embeddings_model,
            persist_directory=persist_directory,
            collection_name=collection_name,
        )

        self.vector_store_classes = Chroma(
            embedding_function=self.embeddings_model,
            persist_directory=persist_directory,
            collection_name=f"{collection_name}_classes",
        )

        self.persist_directory = persist_directory

    @override
    def ingest(self, **kwargs: Any) -> None:
        """
        Ingests documents, splits them into chunks, and indexes them in the vector store.

        Args:
            file_extension (str): File extension pattern (e.g., '*.txt') for document loading.
            data_path (str): Path to the directory containing the documents.
        """
        data_path = kwargs.get("data_path", "")
        if not data_path:
            raise ValueError("data_path is required for document ingestion")

        docs = self.load_docs(data_path)
        if not docs:
            raise ValueError(f"No documents were loaded from {data_path}")

        all_splits = self.split_docs(docs)
        self.add_index(all_splits)
        logging.info("🎉 All documents ingested and indexed")

    @override
    def similarity_search(
        self, question: str, k: int = 5, filter: Dict[str, str] = None
    ) -> List[Any]:
        """
        Performs a similarity search in the vector store.

        Args:
            question (str): The input query for similarity search.
            k (int, optional): The number of top results to retrieve. Defaults to 2.

        Returns:
            List[Any]: A list of top-k similar documents.
        """
        return self.vector_store.similarity_search(question, k=k, filter=filter)

    @override
    def similarity_search_class(
        self, question: str, k: int = 5, filter: Dict[str, str] = None
    ) -> List[Any]:
        """
        Performs a similarity search in the vector store.

        Args:
            question (str): The input query for similarity search.
            k (int, optional): The number of top results to retrieve. Defaults to 2.

        Returns:
            List[Any]: A list of top-k similar documents.
        """
        return self.vector_store_classes.similarity_search(question, k=k, filter=filter)

    def split_docs(
        self, docs: List[Any], chunk_size: int = 2500, chunk_overlap: int = 250
    ) -> List[Any]:
        """
        Splits documents into smaller chunks for indexing.

        Args:
            docs (List[Any]): List of documents to be split.
            chunk_size (int, optional): Maximum size of each chunk. Defaults to 2000.
            chunk_overlap (int, optional): Overlap size between chunks. Defaults to 100.

        Returns:
            List[Any]: A list of document chunks.
        """
        try:
            logging.info("⏳ Splitting documents...")
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )
            all_splits = text_splitter.split_documents(docs)
            logging.info(f"✅ {len(all_splits)} document splits created")
        except Exception as e:
            logging.warning(f"Error occured during documents processing : {e}")
        return all_splits

    def add_index(self, all_splits: List[Any]) -> None:
        """
        Adds document chunks to the Chroma vector store index.

        Args:
            all_splits (List[Any]): List of document chunks to index.

        Returns:
            Any: Result of the indexing operation.
        """
        logging.info("⏳ Adding documents to index...")
        batch_size = 2500
        for i in range(0, len(all_splits), batch_size):
            batch = all_splits[i : i + batch_size]
            _ = self.vector_store.add_documents(documents=batch)
        logging.info("✅ Documents added to index")

    def load_docs(self, data_path: str) -> List[Document]:
        """
        Loads all documents from a directory, using PyPDFLoader for PDFs
        and DirectoryLoader for other file types.

        Args:
            data_path (str): Path to the directory containing the documents.

        Returns:
            List[Document]: A list of loaded documents.
        """
        all_docs = []

        try:
            logging.info(f"⏳ Loading documents from {data_path}...")

            pdf_files = glob.glob(os.path.join(data_path, "*.pdf"))
            for pdf_file in pdf_files:
                try:
                    loader = PyPDFLoader(pdf_file)
                    pdf_docs = loader.load()
                    for doc in pdf_docs:
                        doc.metadata = {"source": os.path.basename(pdf_file)}
                    all_docs.extend(pdf_docs)
                    logging.info(
                        f"✅ Successfully loaded PDF: {os.path.basename(pdf_file)}"
                    )
                except Exception as e:
                    logging.warning(f"⚠️ Failed to load PDF {pdf_file}: {e}")

            non_pdf_extensions = [
                "*.txt",
                "*.docx",
                "*.html",
                "*.md",
                "*.csv",
                "*.json",
            ]
            for ext in non_pdf_extensions:
                try:
                    if glob.glob(os.path.join(data_path, ext)):
                        loader = DirectoryLoader(path=data_path, glob=ext)
                        other_docs = loader.load()
                        for doc in other_docs:
                            doc.metadata = {
                                "source": os.path.basename(doc.metadata["source"])
                            }
                        all_docs.extend(other_docs)
                        logging.info(f"✅ Successfully loaded {ext} files")
                except Exception as e:
                    logging.warning(f"⚠️ Failed to load {ext} files: {e}")

            logging.info(
                f"✅ Total: {len(all_docs)} documents/pages loaded successfully"
            )

        except Exception as e:
            logging.error(f"❌ Critical error during document loading: {e}")

        return all_docs

    @override
    def ingest_code(self, **kwargs: Any) -> None:
        """
        Ingests code from directory, splits it based on the language, and indexes it.

        Args:
            repo_urls (List[str]): List of directories to ingest.
            branch (str, optional): The branch of each repository to clone. Defaults to "main".
        """
        repos_path = kwargs.get("repos_path", "")
        all_splits = []
        class_entries = []

        class_map = self.get_classes(repos_path)

        for file_path, class_signatures in class_map.items():
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    code = f.read()

                splitter = RecursiveCharacterTextSplitter.from_language(
                    language=self.get_language_from_extension(file_path.split(".")[-1]),
                    chunk_size=2500,
                    chunk_overlap=250,
                )
                splits = splitter.create_documents([code])

                for split in splits:
                    split.metadata = {
                        "source": file_path,
                        "classes": ", ".join(class_signatures),
                    }

                all_splits.extend(splits)

                for class_sig in class_signatures:
                    doc = Document(
                        page_content=class_sig, metadata={"source": file_path}
                    )
                    class_entries.append(doc)

            except Exception as e:
                logging.warning(f"⚠️ Error processing {file_path}: {e}")

        if all_splits:
            self.add_index(all_splits)
            logging.info("✅ Documents indexed")

        if class_entries:
            logging.info("🔎 Indexing class names separately...")
            self.vector_store_classes.add_documents(documents=class_entries)
            logging.info("✅ Classes indexed in separate vector store")

        logging.info("🎉 All code files ingested and indexed")
