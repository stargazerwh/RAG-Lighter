# src/raglight/document_processing/dl_text_processor.py
"""
Deep Learning based text segmentation using ModelScope.
"""

import logging
from typing import List, Dict
from langchain_core.documents import Document
from .document_processor import DocumentProcessor
from ..config.settings import Settings

try:
    from modelscope.outputs import OutputKeys
    from modelscope.pipelines import pipeline
    from modelscope.utils.constant import Tasks
    MODELSCOPE_AVAILABLE = True
except ImportError:
    MODELSCOPE_AVAILABLE = False
    logging.warning("ModelScope not installed. DL segmentation disabled.")


class DLTextProcessor(DocumentProcessor):
    """
    Text processor using deep learning-based document segmentation.
    
    Uses ModelScope's BERT-based Chinese document segmentation model
    for intelligent text chunking based on semantic structure.
    """

    def __init__(self, model_name: str = None):
        """
        Initialize the DL text processor.
        
        Args:
            model_name: ModelScope model name for document segmentation.
                       Defaults to Settings.DL_SEGMENTATION_MODEL
        """
        # Use model from settings if not specified
        self.model_name = model_name or Settings.DL_SEGMENTATION_MODEL
        self.enabled = Settings.DL_SEGMENTATION_ENABLED
        self.fallback = Settings.DL_SEGMENTATION_FALLBACK
        self.pipeline = None
        
        # Only load model if DL segmentation is enabled
        if self.enabled and MODELSCOPE_AVAILABLE:
            try:
                self.pipeline = pipeline(
                    task=Tasks.document_segmentation,
                    model=self.model_name
                )
                logging.info(f"Loaded DL segmentation model: {self.model_name}")
            except Exception as e:
                logging.error(f"Failed to load DL model {self.model_name}: {e}")
                self.pipeline = None
        elif not MODELSCOPE_AVAILABLE and self.enabled:
            logging.warning("DL segmentation enabled but ModelScope not installed")
            self.enabled = False

    def process(
        self, file_path: str, chunk_size: int = None, chunk_overlap: int = None
    ) -> Dict[str, List[Document]]:
        """
        Process text file using deep learning segmentation.
        
        Args:
            file_path: Path to text file
            chunk_size: Ignored (DL model determines segments)
            chunk_overlap: Ignored (DL model determines segments)
            
        Returns:
            Dictionary with chunks and classes
        """
        try:
            # Read file
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()
            except UnicodeDecodeError:
                logging.warning(f"UTF-8 decode failed for {file_path}, trying latin-1.")
                with open(file_path, "r", encoding="latin-1") as f:
                    text = f.read()

            if not text:
                logging.warning(f"File {file_path} is empty.")
                return {"chunks": [], "classes": []}

            # Use DL model for segmentation if enabled and available
            if self.enabled and self.pipeline:
                chunks = self._segment_with_dl(text, file_path)
            elif self.fallback:
                # Fallback to traditional method
                chunks = self._segment_with_traditional(text, file_path)
            else:
                logging.error("DL segmentation failed and fallback disabled")
                return {"chunks": [], "classes": []}

            return {"chunks": chunks, "classes": []}

        except Exception as e:
            logging.error(f"Failed to process text file {file_path}. Error: {e}")
            return {"chunks": [], "classes": []}

    def _segment_with_dl(self, text: str, file_path: str) -> List[Document]:
        """
        Segment text using deep learning model from ModelScope.
        """
        try:
            result = self.pipeline(text)
            segments = result.get(OutputKeys.TEXT, [])
            
            if not segments:
                if self.fallback:
                    return self._segment_with_traditional(text, file_path)
                return [Document(page_content=text, metadata={"source": file_path})]
            
            chunks = []
            for i, segment in enumerate(segments):
                if len(segment.strip()) < 10:
                    continue
                    
                doc = Document(
                    page_content=segment.strip(),
                    metadata={
                        "source": file_path,
                        "segment_index": i,
                        "segmentation_method": "deep_learning",
                        "dl_model": self.model_name
                    }
                )
                chunks.append(doc)
            
            logging.info(f"DL segmentation produced {len(chunks)} chunks using {self.model_name}")
            return chunks
            
        except Exception as e:
            logging.warning(f"DL segmentation failed: {e}")
            if self.fallback:
                return self._segment_with_traditional(text, file_path)
            raise

    def _segment_with_traditional(self, text: str, file_path: str) -> List[Document]:
        """Fallback to traditional RecursiveCharacterTextSplitter."""
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        
        doc = Document(page_content=text, metadata={"source": file_path})
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2500,
            chunk_overlap=250
        )
        chunks = text_splitter.split_documents([doc])
        
        for chunk in chunks:
            chunk.metadata["segmentation_method"] = "traditional"
        
        return chunks


class DLPDFProcessor(DocumentProcessor):
    """
    PDF processor using deep learning-based document segmentation.
    """

    def __init__(self, model_name: str = None):
        """
        Initialize the DL PDF processor.
        
        Args:
            model_name: ModelScope model name. Defaults to Settings.DL_SEGMENTATION_MODEL
        """
        model = model_name or Settings.DL_SEGMENTATION_MODEL
        self.text_processor = DLTextProcessor(model)

    def process(
        self, file_path: str, chunk_size: int = None, chunk_overlap: int = None
    ) -> Dict[str, List[Document]]:
        """Process PDF file using deep learning segmentation."""
        try:
            import fitz

            doc = fitz.open(file_path)
            all_chunks = []

            for page_index, page in enumerate(doc):
                text = page.get_text()
                
                if not text.strip():
                    continue

                # Use DL segmentation for each page
                if self.text_processor.enabled and self.text_processor.pipeline:
                    page_chunks = self.text_processor._segment_with_dl(text, file_path)
                else:
                    page_chunks = self.text_processor._segment_with_traditional(text, file_path)
                
                for chunk in page_chunks:
                    chunk.metadata["page"] = page_index
                
                all_chunks.extend(page_chunks)

            doc.close()

            logging.info(f"DL PDF processing produced {len(all_chunks)} chunks from {file_path}")
            return {"chunks": all_chunks, "classes": []}

        except Exception as e:
            logging.error(f"Failed to process PDF {file_path}. Error: {e}")
            return {"chunks": [], "classes": []}
