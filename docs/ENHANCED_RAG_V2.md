# RAG-Lighter 增强版实现文档

## 🎯 实现概述

本次增强为 RAG-Lighter 添加了两大核心功能：

1. **父子分块 (Parent-Child Chunking)**：小块检索，大块生成
2. **查询改写 (Query Rewriting)**：支持 Direct/HyDE/Subquery/Auto 策略

---

## 📁 文件清单

### 新增文件

| 文件 | 说明 |
|------|------|
| `strategy_selector.py` | 策略选择器，让 LLM 动态选择查询策略 |
| `query_rewriter.py` | 查询改写器，实现 Direct/HyDE/Subquery |

### 修改文件

| 文件 | 修改内容 |
|------|----------|
| `rag_config.py` | 添加父子分块和查询策略配置 |
| `text_processor.py` | 支持 `use_parent_child` 和 `chunk_config` |
| `pdf_processor.py` | 支持 `use_parent_child` 和 `chunk_config` |
| `vector_store.py` | 添加父子分块抽象接口 |
| `chroma.py` | 实现父子分块存储和检索 |
| `rag.py` | 重写，集成策略选择、查询改写、父子分块 |
| `builder.py` | 支持通过 RAGConfig 构建 |
| `rag/__init__.py` | 导出新增类 |

---

## ⚙️ 配置说明

### RAGConfig 新增字段

```python
@dataclass
class RAGConfig:
    # === 父子分块配置 ===
    use_parent_child_chunking: bool = False  # 是否启用
    parent_chunk_size: int = 2000            # 父块大小
    parent_chunk_overlap: int = 200          # 父块重叠
    child_chunk_size: int = 400              # 子块大小
    child_chunk_overlap: int = 50            # 子块重叠
    
    # === 查询改写策略 ===
    # "Auto" = LLM 动态选择
    # "Direct" = 直接查询
    # "HyDE" = 假设文档
    # "Subquery" = 子查询分解
    query_rewrite_strategy: str = "Auto"
```

---

## 🚀 使用方式

### 基础用法

```python
from raglight.config.rag_config import RAGConfig
from raglight.config.settings import Settings
from raglight.rag.builder import Builder

# 配置
config = RAGConfig(
    llm="llama3.1:8b",
    k=3,
    use_parent_child_chunking=True,
    parent_chunk_size=2000,
    child_chunk_size=400,
    query_rewrite_strategy="Auto"
)

# 构建
rag = (
    Builder()
    .with_embeddings(Settings.HUGGINGFACE, model_name="all-MiniLM-L6-v2")
    .with_vector_store(Settings.CHROMA, collection_name="my_app")
    .with_llm(Settings.OLLAMA, model_name="llama3.1:8b")
    .build_rag(config=config)
)

# 使用
response = rag.generate("你的问题？")
print(f"使用策略: {rag.get_last_strategy()}")
```

### 配置组合示例

| 场景 | 父子分块 | 查询策略 | 说明 |
|------|---------|---------|------|
| 基础 | False | Direct | 标准 RAG |
| 精准检索 | True | Direct | 小块检索，大块生成 |
| 抽象问题 | False | HyDE | 生成假设文档辅助检索 |
| 复杂问题 | False | Subquery | 分解为子查询 |
| 全自动 | True | Auto | 推荐配置 |

---

## 🔍 核心流程

### 完整流程（Auto + 父子分块）

```
用户问题
    ↓
[StrategySelector] ──▶ LLM 分析选择策略 (Direct/HyDE/Subquery)
    ↓
[QueryRewriter] ──▶ 根据策略改写查询
    ├─ Direct: [原查询]
    ├─ HyDE: [假设文档, 原查询]
    └─ Subquery: [子查询1, 子查询2, ...]
    ↓
[多查询检索]
    ├─ 父子分块: similarity_search_parent_child()
    │   ├─ 子块向量检索 (k*10)
    │   ├─ CrossEncoder 重排序子块 (如果有)
    │   ├─ 按重排序后顺序获取父块
    │   └─ 合并重复父块
    └─ 标准: similarity_search()
    ↓
[合并结果] ──▶ 按检索顺序直接拼接（不排序不截断）
    ↓
[LLM 生成] ──▶ 最终答案
```

### Reranker 在父子分块中的位置

```
标准 RAG 重排序:
  检索 chunks ──▶ CrossEncoder 重排序 chunks ──▶ 返回 Top-k chunks

父子分块 RAG 重排序:
  检索子块 ──▶ CrossEncoder 重排序子块 ──▶ 按子块顺序获取父块 ──▶ 合并重复父块 ──▶ 返回 Top-k 父块

关键区别：
- 标准模式：重排序对象是检索结果（chunks）
- 父子分块：重排序对象是子块，但返回的是父块
- 优势：细粒度重排序 + 粗粒度上下文
```

---

## 📦 OSS 下载地址

**基础 URL**: `https://xl12442444.oss-cn-hangzhou.aliyuncs.com/RAG-Lighter/v2/`

| 文件 | URL |
|------|-----|
| rag_config.py | /v2/rag_config.py |
| strategy_selector.py | /v2/strategy_selector.py |
| query_rewriter.py | /v2/query_rewriter.py |
| text_processor.py | /v2/text_processor.py |
| pdf_processor.py | /v2/pdf_processor.py |
| vector_store.py | /v2/vector_store.py |
| chroma.py | /v2/chroma.py |
| rag.py | /v2/rag.py |
| builder.py | /v2/builder.py |
| enhanced_rag_example.py | /v2/enhanced_rag_example.py |

---

## 🔄 与之前父子分块的区别

| 特性 | 之前的独立实现 | 现在的集成实现 |
|------|---------------|---------------|
| 使用方式 | ParentChildRAG 独立类 | 通过 RAGConfig 开关 |
| 代码结构 | 独立文件 | 集成到核心 RAG |
| 配置方式 | 手动组装 | 统一 RAGConfig |
| 查询改写 | 不支持 | 支持 |
| 向后兼容 | 需单独使用 | 默认关闭，向后兼容 |

---

## ⚠️ 注意事项

1. **Breaking Change**: 需要更新现有代码，使用新的 RAGConfig 方式配置
2. **依赖**: 需要安装 `chromadb`, `langchain`, `langgraph`
3. **性能**: Auto 策略会增加一次 LLM 调用（用于策略选择）
4. **存储**: 父子分块会增加约 20% 存储开销

---

**实现完成！** 🎉
