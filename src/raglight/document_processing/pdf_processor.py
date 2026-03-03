import logging
import fitz
from typing import List, Dict, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from .document_processor import DocumentProcessor


class PDFProcessor(DocumentProcessor):
    def process(
        self, 
        file_path: str, 
        chunk_size: int, 
        chunk_overlap: int,
        use_parent_child: bool = False,
        chunk_config: Optional[Dict] = None
    ) -> Dict[str, List[Document]]:
        """
        处理 PDF 文件
        
        Args:
            file_path: 文件路径
            chunk_size: 切块大小
            chunk_overlap: 切块重叠
            use_parent_child: 是否使用父子分块
            chunk_config: 父子分块配置字典
            
        Returns:
            标准模式: {"chunks": [...], "classes": []}
            父子模式: {"parents": [...], "children": [...], "classes": []}
        """
        try:
            doc = fitz.open(file_path)
            raw_documents = []

            for page_index, page in enumerate(doc):
                blocks = page.get_text("dict")["blocks"]
                page_text_parts = []

                for block in blocks:
                    if block["type"] == 0:
                        for line in block["lines"]:
                            for span in line["spans"]:
                                page_text_parts.append(span["text"])
                            page_text_parts.append("\n")
                        page_text_parts.append("\n")

                full_page_text = "".join(page_text_parts).strip()

                if full_page_text:
                    document = Document(
                        page_content=full_page_text,
                        metadata={"source": file_path, "page": page_index},
                    )
                    raw_documents.append(document)

            doc.close()

            if not raw_documents:
                logging.warning(f"PyMuPDF returned no text for {file_path}.")
                if use_parent_child:
                    return {"parents": [], "children": [], "classes": []}
                return {"chunks": [], "classes": []}

            # 父子分块模式
            if use_parent_child:
                from .parent_child_processor import ParentChildProcessor
                
                config = chunk_config or {}
                parent_size = config.get("parent_chunk_size", chunk_size * 4)
                parent_overlap = config.get("parent_chunk_overlap", chunk_overlap)
                child_size = config.get("child_chunk_size", chunk_size)
                child_overlap = config.get("child_chunk_overlap", chunk_overlap // 2)
                
                processor = ParentChildProcessor(
                    parent_chunk_size=parent_size,
                    parent_chunk_overlap=parent_overlap,
                    child_chunk_size=child_size,
                    child_chunk_overlap=child_overlap
                )
                
                # 合并所有页面文本
                full_text = "\n\n".join([doc.page_content for doc in raw_documents])
                result = processor.process_text(full_text, source=file_path)
                result["classes"] = []
                return result

            # 标准分块模式
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )
            chunks = text_splitter.split_documents(raw_documents)

            for chunk in chunks:
                chunk.metadata["source"] = file_path

            return {"chunks": chunks, "classes": []}

        except Exception as e:
            logging.error(f"Failed to process PDF {file_path}. Error: {e}")
            if use_parent_child:
                return {"parents": [], "children": [], "classes": []}
            return {"chunks": [], "classes": []}
