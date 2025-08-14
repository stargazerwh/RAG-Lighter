import logging
from typing import List, Dict
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from .document_processor import DocumentProcessor


class TextProcessor(DocumentProcessor):
    def process(
        self, file_path: str, chunk_size: int, chunk_overlap: int
    ) -> Dict[str, List[Document]]:
        try:
            loader = UnstructuredFileLoader(file_path, mode="single")
            docs = loader.load()

            if not docs:
                logging.warning(
                    f"UnstructuredFileLoader returned no documents for {file_path}."
                )
                return {"chunks": [], "classes": []}

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )
            chunks = text_splitter.split_documents(docs)

            return {"chunks": chunks, "classes": []}
        except Exception as e:
            logging.error(f"Failed to process text file {file_path}. Error: {e}")
            return {"chunks": [], "classes": []}
