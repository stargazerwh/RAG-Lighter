"""
Parent-Child Chunking Processor
父子切块处理器 - 小块用于检索，大块用于生成
"""

import hashlib
import logging
from typing import List, Dict, Tuple
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from .document_processor import DocumentProcessor

logger = logging.getLogger(__name__)


class ParentChildProcessor(DocumentProcessor):
    """
    Parent-Child Chunking Processor
    
    实现父子切块策略：
    - Parent (父块): 大块文本，用于提供给 LLM 的完整上下文
    - Child (子块): 小块文本，用于精准向量检索
    
    工作流程：
    1. 先将文档切成大块的父块 (parent)
    2. 每个父块内部再切成小的子块 (child)
    3. 子块通过 parent_id 关联到父块
    4. 检索时：匹配子块 → 返回对应父块
    """

    def __init__(
        self,
        parent_chunk_size: int = 2000,
        parent_chunk_overlap: int = 200,
        child_chunk_size: int = 400,
        child_chunk_overlap: int = 50,
        length_function=len,
    ):
        """
        初始化父子切块处理器

        Args:
            parent_chunk_size: 父块大小（字符数）
            parent_chunk_overlap: 父块重叠大小
            child_chunk_size: 子块大小（字符数），用于检索
            child_chunk_overlap: 子块重叠大小
            length_function: 长度计算函数
        """
        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=parent_chunk_size,
            chunk_overlap=parent_chunk_overlap,
            length_function=length_function,
            separators=["\n\n", "\n", ".", "!", "?", ";", " ", ""]
        )
        
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=child_chunk_size,
            chunk_overlap=child_chunk_overlap,
            length_function=length_function,
            separators=["\n\n", "\n", ".", "!", "?", ";", " ", ""]
        )
        
        self.parent_chunk_size = parent_chunk_size
        self.child_chunk_size = child_chunk_size
        
        logger.info(
            f"ParentChildProcessor initialized: "
            f"parent={parent_chunk_size}/{parent_chunk_overlap}, "
            f"child={child_chunk_size}/{child_chunk_overlap}"
        )

    def process(
        self, 
        file_path: str, 
        chunk_size: int = None,  # 兼容接口，实际使用 child_chunk_size
        chunk_overlap: int = None
    ) -> Dict[str, List[Document]]:
        """
        处理文件，生成父子切块

        Args:
            file_path: 文件路径
            chunk_size: 保留参数（兼容接口）
            chunk_overlap: 保留参数（兼容接口）

        Returns:
            Dict with 'parents' and 'children' keys
        """
        try:
            # 读取文件内容
            text = self._read_file(file_path)
            if not text:
                return {"parents": [], "children": []}
            
            # 生成父子切块
            return self._create_parent_child_chunks(text, file_path)
            
        except Exception as e:
            logger.error(f"Failed to process {file_path}: {e}")
            return {"parents": [], "children": []}

    def _read_file(self, file_path: str) -> str:
        """读取文件内容"""
        encodings = ['utf-8', 'latin-1', 'gbk', 'gb2312']
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
        
        # 如果都失败，使用 errors='ignore'
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()

    def _create_parent_child_chunks(
        self, 
        text: str, 
        source: str
    ) -> Dict[str, List[Document]]:
        """
        创建父子切块
        
        策略：
        1. 先用大窗口切父块
        2. 每个父块内部用小窗口切子块
        3. 子块继承父块的 metadata
        """
        # Step 1: 创建父块
        parent_docs = self.parent_splitter.create_documents(
            [text], 
            metadatas=[{"source": source}]
        )
        
        all_parents = []
        all_children = []
        
        for parent_idx, parent_doc in enumerate(parent_docs):
            # 为父块生成唯一 ID
            parent_id = self._generate_doc_id(parent_doc.page_content, source, parent_idx)
            
            # 设置父块 metadata
            parent_doc.metadata.update({
                "doc_id": parent_id,
                "doc_type": "parent",
                "chunk_index": parent_idx,
                "chunk_size": len(parent_doc.page_content),
                "is_parent": True
            })
            all_parents.append(parent_doc)
            
            # Step 2: 在父块内部切子块
            child_docs = self.child_splitter.split_documents([parent_doc])
            
            for child_idx, child_doc in enumerate(child_docs):
                child_doc.metadata.update({
                    "parent_id": parent_id,
                    "doc_type": "child",
                    "parent_chunk_index": parent_idx,
                    "child_chunk_index": child_idx,
                    "chunk_size": len(child_doc.page_content),
                    "is_parent": False,
                    "source": source
                })
                all_children.append(child_doc)
        
        logger.info(
            f"Created {len(all_parents)} parents, {len(all_children)} children "
            f"from {source}"
        )
        
        return {
            "parents": all_parents,
            "children": all_children
        }

    def _generate_doc_id(self, content: str, source: str, index: int) -> str:
        """生成文档唯一 ID"""
        hash_input = f"{source}:{index}:{content[:100]}"
        return hashlib.md5(hash_input.encode()).hexdigest()[:16]

    def process_text(
        self, 
        text: str, 
        source: str = "inline"
    ) -> Dict[str, List[Document]]:
        """
        直接处理文本（不通过文件）
        
        Args:
            text: 输入文本
            source: 来源标识
            
        Returns:
            Dict with 'parents' and 'children'
        """
        if not text:
            return {"parents": [], "children": []}
        
        return self._create_parent_child_chunks(text, source)


class HierarchicalChunker:
    """
    层级切块器 - 支持更多层级（可选扩展）
    
    例如：Section -> Paragraph -> Sentence
    """
    
    def __init__(
        self,
        level_sizes: List[int] = [4000, 1000, 250],
        level_overlaps: List[int] = [400, 100, 25]
    ):
        """
        多级切块
        
        Args:
            level_sizes: 每级切块大小 [level_0, level_1, level_2...]
            level_overlaps: 每级重叠大小
        """
        self.levels = len(level_sizes)
        self.splitters = []
        
        for size, overlap in zip(level_sizes, level_overlaps):
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=size,
                chunk_overlap=overlap,
                separators=["\n\n", "\n", ".", "!", "?", ";", " ", ""]
            )
            self.splitters.append(splitter)
    
    def process(self, text: str, source: str = "inline") -> Dict[int, List[Document]]:
        """
        多级切块
        
        Returns:
            Dict[level_index, List[Document]]
        """
        results = {}
        current_chunks = [Document(page_content=text, metadata={"source": source})]
        
        for level, splitter in enumerate(self.splitters):
            next_chunks = []
            for doc in current_chunks:
                split_docs = splitter.split_documents([doc])
                for i, sd in enumerate(split_docs):
                    sd.metadata["level"] = level
                    sd.metadata["index"] = i
                    if level > 0:
                        sd.metadata["parent_level"] = level - 1
                next_chunks.extend(split_docs)
            
            results[level] = next_chunks
            current_chunks = next_chunks
        
        return results
