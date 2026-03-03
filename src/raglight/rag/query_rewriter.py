"""
Query Rewriter - 查询改写器
支持 Direct / HyDE / Subquery 三种策略
"""

import logging
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from ..llm.llm import LLM

logger = logging.getLogger(__name__)


class QueryRewriter:
    """
    查询改写器
    
    根据选择的策略，将用户查询改写为适合检索的形式：
    - Direct: 保持原样
    - HyDE: 生成假设文档 + 保留原查询
    - Subquery: 分解为多个子查询
    """
    
    def __init__(self, llm: "LLM"):
        """
        初始化查询改写器
        
        Args:
            llm: 大语言模型实例
        """
        self.llm = llm
        logger.info("QueryRewriter initialized")
    
    def rewrite(self, query: str, strategy: str) -> List[str]:
        """
        改写查询
        
        Args:
            query: 原始查询
            strategy: 策略名称 (Direct/HyDE/Subquery)
            
        Returns:
            改写后的查询列表（多个查询都会被执行检索）
        """
        if strategy == "Direct":
            return self._direct_rewrite(query)
        elif strategy == "HyDE":
            return self._hyde_rewrite(query)
        elif strategy == "Subquery":
            return self._subquery_rewrite(query)
        else:
            logger.warning(f"Unknown strategy: {strategy}, fallback to Direct")
            return self._direct_rewrite(query)
    
    def _direct_rewrite(self, query: str) -> List[str]:
        """
        Direct 策略：直接返回原查询
        
        Returns:
            [原始查询]
        """
        logger.debug(f"Direct strategy: {query}")
        return [query]
    
    def _hyde_rewrite(self, query: str) -> List[str]:
        """
        HyDE 策略：生成假设文档 + 保留原查询
        
        生成一段假设包含答案的文档，用于增强检索。
        同时保留原查询，两者都会进行检索。
        
        Returns:
            [假设文档, 原始查询]  # 两者都检索
        """
        prompt = f"""基于以下问题，写一段详细的技术文档段落，这段文档应该包含问题的答案。
文档应该像知识库中的真实内容一样，专业、详细、包含关键信息。

问题：{query}

请直接输出文档内容，不要添加标题或额外说明："""
        
        try:
            hypothetical_doc = self.llm.generate({"question": prompt})
            hypothetical_doc = hypothetical_doc.strip()
            
            logger.info(f"HyDE generated document: {len(hypothetical_doc)} chars")
            
            # 返回假设文档和原查询（两者都检索）
            return [hypothetical_doc, query]
            
        except Exception as e:
            logger.error(f"HyDE generation failed: {e}, fallback to original query")
            return [query]
    
    def _subquery_rewrite(self, query: str) -> List[str]:
        """
        Subquery 策略：让 LLM 决定分解为多少个子查询
        
        根据问题复杂度，自动决定分解为 2-5 个子查询。
        
        Returns:
            [子查询1, 子查询2, ...]
        """
        prompt = f"""你是一个查询分解专家。请将以下复杂问题分解为多个简单的子查询。

要求：
1. 分析问题的复杂度，决定分解为 2-5 个子查询
2. 每个子查询应该独立可回答
3. 子查询组合起来应该能回答原问题
4. 只输出子查询列表，每行一个，不要序号

问题：{query}

子查询列表："""
        
        try:
            response = self.llm.generate({"question": prompt})
            subqueries = self._parse_subqueries(response)
            
            if len(subqueries) < 2:
                logger.warning("Subquery decomposition returned too few queries, using original")
                return [query]
            
            logger.info(f"Subquery decomposition: {len(subqueries)} queries")
            for i, sq in enumerate(subqueries, 1):
                logger.debug(f"  [{i}] {sq}")
            
            return subqueries
            
        except Exception as e:
            logger.error(f"Subquery decomposition failed: {e}, fallback to original query")
            return [query]
    
    def _parse_subqueries(self, text: str) -> List[str]:
        """
        解析 LLM 返回的子查询列表
        
        Args:
            text: LLM 返回的文本
            
        Returns:
            子查询列表
        """
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        subqueries = []
        for line in lines:
            # 去掉常见的前缀
            prefixes = ['-', '*', '•', '1.', '2.', '3.', '4.', '5.', 
                       '1)', '2)', '3)', '4)', '5)',
                       '子查询', '查询']
            
            clean_line = line
            for prefix in prefixes:
                if clean_line.startswith(prefix):
                    clean_line = clean_line[len(prefix):].strip()
                    break
            
            # 去掉序号前缀 (如 "1. " 或 "1 ")
            import re
            clean_line = re.sub(r'^\d+[\.\)\s]+', '', clean_line)
            
            if clean_line and len(clean_line) > 10:  # 过滤太短的行
                subqueries.append(clean_line)
        
        return subqueries
