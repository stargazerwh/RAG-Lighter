import logging
from typing import List, Dict, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from .document_processor import DocumentProcessor


class TextProcessor(DocumentProcessor):
    def process(
        self, 
        file_path: str, 
        chunk_size: int, 
        chunk_overlap: int,
        use_parent_child: bool = False,
        chunk_config: Optional[Dict] = None
    ) -> Dict[str, List[Document]]:
        """
        处理文本文件
        
        Args:
            file_path: 文件路径
            chunk_size: 切块大小（标准模式使用，父子模式作为子块大小参考）
            chunk_overlap: 切块重叠
            use_parent_child: 是否使用父子分块
            chunk_config: 父子分块配置字典，包含 parent_chunk_size, parent_chunk_overlap 等
            
        Returns:
            标准模式: {"chunks": [...], "classes": []}
            父子模式: {"parents": [...], "children": [...], "classes": []}
        """
        try:
            # 读取文件
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()
            except UnicodeDecodeError:
                logging.warning(f"UTF-8 decode failed for {file_path}, trying latin-1.")
                with open(file_path, "r", encoding="latin-1") as f:
                    text = f.read()

            if not text:
                logging.warning(f"File {file_path} is empty.")
                return {"chunks": [], "classes": []} if not use_parent_child else {"parents": [], "children": [], "classes": []}

            # 父子分块模式
            if use_parent_child:
                from .parent_child_processor import ParentChildProcessor
                
                # 从 chunk_config 获取参数，或使用默认值
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
                
                result = processor.process(file_path, chunk_size, chunk_overlap)
                result["classes"] = []  # 保持返回格式一致
                return result
            
            # 标准分块模式
            doc = Document(page_content=text, metadata={"source": file_path})

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )
            chunks = text_splitter.split_documents([doc])

            return {"chunks": chunks, "classes": []}

        except Exception as e:
            logging.error(f"Failed to process text file {file_path}. Error: {e}")
            if use_parent_child:
                return {"parents": [], "children": [], "classes": []}
            return {"chunks": [], "classes": []}
