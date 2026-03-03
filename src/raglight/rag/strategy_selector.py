"""
Strategy Selector - 查询策略智能选择器
让 LLM 根据问题特征动态选择最优检索策略
"""

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..llm.llm import LLM

logger = logging.getLogger(__name__)


class StrategySelector:
    """
    查询策略选择器
    
    角色设定：你是 RAG 系统的策略顾问，专门分析用户查询的特征，
    选择最适合的检索策略以获得最佳回答质量。
    """
    
    # 策略说明
    STRATEGIES = {
        "Direct": {
            "description": "直接检索",
            "适用场景": [
                "问题简单明确，关键词清晰",
                "需要查找具体事实、定义、数据",
                "单一路径即可找到答案"
            ],
            "示例": [
                "Python 的列表推导式语法是什么？",
                "2024年巴黎奥运会举办时间？",
                "Transformer 论文的发表年份？"
            ]
        },
        "HyDE": {
            "description": "假设文档检索 (Hypothetical Document Embedding)",
            "适用场景": [
                "问题抽象、概念性、开放式",
                "需要理解意图而非匹配关键词",
                "原查询可能无法直接匹配文档"
            ],
            "示例": [
                "如何设计一个高可用的微服务架构？",
                "解释量子计算对密码学的影响",
                "RAG 系统的最佳实践有哪些？"
            ]
        },
        "Subquery": {
            "description": "子查询分解",
            "适用场景": [
                "问题复杂，包含多个条件或维度",
                "需要组合多个信息源才能回答",
                "问题可以拆分为独立的子问题"
            ],
            "示例": [
                "比较 Python 和 JavaScript 在异步处理和内存管理方面的差异",
                "Transformer 和 RNN 在训练效率、长文本处理、并行化方面各有什么优缺点？",
                "如何在保证系统性能的同时实现数据一致性和故障恢复？"
            ]
        }
    }
    
    def __init__(self, llm: "LLM"):
        """
        初始化策略选择器
        
        Args:
            llm: 大语言模型实例，用于分析查询
        """
        self.llm = llm
        logger.info("StrategySelector initialized")
    
    def select(self, query: str) -> str:
        """
        为给定查询选择最佳策略
        
        Args:
            query: 用户查询
            
        Returns:
            策略名称: "Direct" | "HyDE" | "Subquery"
        """
        prompt = self._build_prompt(query)
        
        try:
            response = self.llm.generate(prompt)
            strategy = self._parse_response(response)
            
            logger.info(f"Query: '{query[:50]}...' -> Strategy: {strategy}")
            return strategy
            
        except Exception as e:
            logger.error(f"Strategy selection failed: {e}, fallback to Direct")
            return "Direct"
    
    def _build_prompt(self, query: str) -> str:
        """构建策略选择提示词"""
        
        prompt = f"""【角色设定】
你是 RAG (检索增强生成) 系统的策略顾问。你的任务是根据用户查询的特征，
选择最适合的检索策略，以获得最佳的回答质量。

【可选策略】

1. Direct (直接检索)
   - 适用：问题简单明确，关键词清晰，需要查找具体事实
   - 示例：
     * "Python 的列表推导式语法是什么？"
     * "2024年巴黎奥运会举办时间？"
     * "Transformer 论文发表年份？"

2. HyDE (假设文档检索)
   - 适用：问题抽象、概念性，需要理解意图而非匹配关键词
   - 示例：
     * "如何设计一个高可用的微服务架构？"
     * "解释量子计算对密码学的影响"
     * "RAG 系统的最佳实践有哪些？"

3. Subquery (子查询分解)
   - 适用：问题复杂多维度，需要组合多个信息源
   - 示例：
     * "比较 Python 和 JavaScript 在异步处理和内存管理方面的差异"
     * "Transformer 和 RNN 在训练效率、长文本处理、并行化方面各有什么优缺点？"

【任务】
分析以下用户查询，选择最合适的策略。
只需要返回策略名称 (Direct/HyDE/Subquery)，不要任何解释。

【用户查询】
{query}

【输出】"""
        
        return {"question": prompt}
    
    def _parse_response(self, response: str) -> str:
        """解析 LLM 返回的策略名称"""
        response = response.strip()
        
        # 直接匹配
        for strategy in ["Direct", "HyDE", "Subquery"]:
            if strategy in response:
                return strategy
        
        # 大小写不敏感匹配
        response_lower = response.lower()
        if "direct" in response_lower:
            return "Direct"
        elif "hyde" in response_lower or "hypothetical" in response_lower:
            return "HyDE"
        elif "subquery" in response_lower or "sub-query" in response_lower:
            return "Subquery"
        
        # 默认 fallback
        logger.warning(f"Could not parse strategy from: {response}, using Direct")
        return "Direct"
    
    def explain(self, query: str, selected_strategy: str) -> str:
        """
        解释为什么选择该策略（用于调试）
        
        Args:
            query: 用户查询
            selected_strategy: 已选择的策略
            
        Returns:
            解释文本
        """
        strategy_info = self.STRATEGIES.get(selected_strategy, {})
        
        explanation = f"""
策略选择解释:
- 查询: {query[:100]}...
- 选择策略: {selected_strategy} ({strategy_info.get('description', 'N/A')})
- 适用场景: {', '.join(strategy_info.get('适用场景', []))}
"""
        return explanation
