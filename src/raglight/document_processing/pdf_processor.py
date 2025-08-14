import logging
from typing import List, Dict
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from .document_processor import DocumentProcessor


class PDFProcessor(DocumentProcessor):
    def process(
        self, file_path: str, chunk_size: int, chunk_overlap: int
    ) -> Dict[str, List[Document]]:
        try:
            loader = PyPDFLoader(file_path)
            pages = loader.load()

            if not pages:
                logging.warning(f"PyPDFLoader returned no pages for {file_path}.")
                return {"chunks": [], "classes": []}

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )
            chunks = text_splitter.split_documents(pages)

            for chunk in chunks:
                chunk.metadata["source"] = file_path

            return {"chunks": chunks, "classes": []}
        except Exception as e:
            logging.error(f"Failed to process PDF {file_path}. Error: {e}")
            return {"chunks": [], "classes": []}
