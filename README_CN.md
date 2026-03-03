# RAG-Lighter

轻量级模块化 RAG (检索增强生成) 框架

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## 特性

- **模块化设计** - 易于扩展和定制
- **多 LLM 提供商** - 支持多种 AI 模型
- **多向量数据库** - 支持 ChromaDB、Milvus
- **Agentic RAG** - 支持工具调用的智能检索
- **多模态支持** - 文本、代码和图像处理
- **RAG 评估** - 使用 RAGAS 指标评估质量
- **训练数据生成** - 从文档自动生成问答对
- **本地与云端** - 本地 Ollama 或云端 API
- **🔥 父子分块** - 小块检索精准，大块生成完整
- **🔥 查询改写** - 支持 Direct/HyDE/Subquery/Auto 策略，LLM 动态选择

## 支持的 LLM 提供商

| 提供商 | 状态 | 说明 |
|--------|------|------|
| **OpenAI** | ✅ 支持 | GPT-4, GPT-3.5 等 |
| **Ollama** | ✅ 支持 | 本地模型 |
| **Mistral** | ✅ 支持 | Mistral AI API |
| **Google Gemini** | ✅ 支持 | Gemini Pro 等 |
| **LM Studio** | ✅ 支持 | 本地 GUI 管理 |
| **Kimi (月之暗面)** | ✅ 支持 | OpenAI 兼容 API |
| **DeepSeek** | ✅ 支持 | 支持推理模型 |

## 支持的向量数据库

| 向量数据库 | 状态 | 说明 |
|------------|------|------|
| **ChromaDB** | ✅ 支持 | 默认，本地/远程 |
| **Milvus** | ✅ 支持 | 支持向量索引 (IVF_FLAT, HNSW 等) |

## 支持的嵌入模型

| 模型 | 状态 | 说明 |
|------|------|------|
| **Sentence Transformers** | ✅ 支持 | all-MiniLM-L6-v2 等 |
| **BGE-M3** | ✅ 支持 | 多语言，1024 维 |
| **OpenAI** | ✅ 支持 | text-embedding-ada-002 |
| **Google Gemini** | ✅ 支持 | gemini-embedding-001 |

## 安装

```bash
pip install raglight
```

或从源码安装：

```bash
git clone https://github.com/stargazerwh/RAG-lightweight.git
cd RAG-lightweight
pip install -e .
```

## 快速开始

### 基础用法

```python
from raglight.rag.builder import Builder
from raglight.config.settings import Settings

# 创建 RAG 管道
rag = (
    Builder()
    .with_embeddings(Settings.HUGGINGFACE, model_name="all-MiniLM-L6-v2")
    .with_vector_store(Settings.CHROMA, collection_name="my_docs")
    .with_llm(Settings.OLLAMA, model_name="llama3.1:8b")
    .build_rag(k=5)
)

# 索引文档
rag.vector_store.ingest("./documents")

# 查询
response = rag.generate("主要内容是什么？")
print(response)
```

### 🔥 父子分块 (推荐)

小块用于精准检索，大块用于完整生成：

```python
from raglight.config.rag_config import RAGConfig

# 配置父子分块
config = RAGConfig(
    llm="llama3.1:8b",
    provider=Settings.OLLAMA,
    use_parent_child_chunking=True,  # 启用父子分块
    parent_chunk_size=2000,    # 父块：给 LLM 的完整上下文
    child_chunk_size=400,      # 子块：用于向量检索
    query_rewrite_strategy="Auto"  # 自动选择策略
)

rag = (
    Builder()
    .with_embeddings(Settings.HUGGINGFACE, model_name="all-MiniLM-L6-v2")
    .with_vector_store(Settings.CHROMA, collection_name="pc_docs")
    .with_llm(Settings.OLLAMA, model_name="llama3.1:8b")
    .with_cross_encoder(Settings.HUGGINGFACE, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
    .build_rag(config=config)
)

# Cross-encoder 重排序子块，返回父块
response = rag.generate("你的复杂问题？")
print(f"使用策略: {rag.get_last_strategy()}")  # Direct/HyDE/Subquery
```

### 🔥 查询改写策略

```python
from raglight.config.rag_config import RAGConfig

config = RAGConfig(
    llm="llama3.1:8b",
    provider=Settings.OLLAMA,
    query_rewrite_strategy="HyDE"  # 或 "Direct", "Subquery", "Auto"
)

rag = Builder().with_embeddings(...).with_vector_store(...).with_llm(...).build_rag(config=config)

# HyDE: 生成假设文档辅助检索
# Subquery: 将复杂问题分解为子查询
# Auto: LLM 动态选择最佳策略
```

### 查询策略对比

| 策略 | 适用场景 | 说明 |
|------|----------|------|
| **Direct** | 简单明确的问题 | 直接查询 |
| **HyDE** | 抽象/概念性问题 | 生成假设答案文档 |
| **Subquery** | 复杂多条件问题 | 分解为子查询 |
| **Auto** | 未知类型问题 | LLM 动态选择策略 |

### 使用 Kimi (月之暗面)

```python
from raglight.llm import KimiModel

kimi = KimiModel(model_name="kimi-k2.5")
response = kimi.generate({"question": "你好，请介绍一下自己"})
print(response)
```

### 使用 DeepSeek

```python
from raglight.llm import DeepSeekModel

# 标准对话模型
deepseek = DeepSeekModel(model_name="deepseek-chat")
response = deepseek.generate({"question": "解释量子计算"})

# 推理模型 (R1)
r1 = DeepSeekModel(model_name="deepseek-reasoner")
result = r1.generate_with_thinking({"question": "解决这个复杂问题..."})
print(result["reasoning"])  # 推理过程
print(result["answer"])     # 最终答案
```

### 使用 Milvus 向量数据库

```python
from raglight.vectorstore import MilvusVS
from raglight.embeddings import HuggingfaceEmbeddingsModel

# 初始化 Milvus 并配置向量索引
embeddings = HuggingfaceEmbeddingsModel("all-MiniLM-L6-v2")

# 本地 Milvus Lite
vector_store = MilvusVS(
    collection_name="my_collection",
    embeddings_model=embeddings,
    persist_directory="./milvus_db",  # Milvus Lite 模式
    index_type="HNSW",  # 或 "IVF_FLAT", "IVF_SQ8"
    metric_type="COSINE"  # 或 "L2", "IP"
)

# 或连接 Milvus 服务器
vector_store = MilvusVS(
    collection_name="my_collection",
    embeddings_model=embeddings,
    host="localhost",
    port=19530,
    index_type="HNSW",
    metric_type="COSINE"
)
```

### 使用 BGE-M3 嵌入模型

```python
from raglight.embeddings import BgeM3EmbeddingsModel

# BGE-M3 多语言检索
embeddings = BgeM3EmbeddingsModel(
    model_name="BAAI/bge-m3",
    use_fp16=True  # 使用半精度加速推理
)

# 或使用 BGE-Large
embeddings = BgeM3EmbeddingsModel("BAAI/bge-large-en-v1.5")
```

### RAG 评估 (RAGAS)

```python
from raglight.evaluation import RAGASEvaluator, RAGResult

# 初始化评估器
evaluator = RAGASEvaluator()

# 创建 RAG 结果
result = RAGResult(
    question="主要内容是什么？",
    answer="文档讨论了...",
    contexts=retrieved_documents
)

# 评估
scores = evaluator.evaluate(result)
print(f"忠实度 (Faithfulness): {scores.faithfulness}")
print(f"答案相关性 (Answer Relevancy): {scores.answer_relevancy}")
print(f"上下文相关性 (Context Relevancy): {scores.context_relevancy}")
print(f"上下文召回率 (Context Recall): {scores.context_recall}")

# 批量评估
results = [result1, result2, result3]
report = evaluator.generate_report(results)
```

### 训练数据生成

从文档自动生成问答对，用于微调或评估：

```python
from raglight.embeddings.build.traindata import load_corpus, documents_to_qa
from raglight.llm import OpenAIModel

# 加载并分块文档
docs = load_corpus(["train.pdf"], chunk_size=512, verbose=True)

# 使用 LLM 生成问答对
llm = OpenAIModel(model_name="gpt-4")
qa_df = documents_to_qa(docs, llm=llm, verbose=True)

# 保存用于训练
qa_df.to_csv("train_qa.csv", index=False)
```

### Agentic RAG

```python
from raglight.agent import Agent

agent = Agent(rag=rag)
response = agent.chat("在我的文档中查找关于 X 的信息")
```

## 环境变量

创建 `.env` 文件：

```bash
# OpenAI
OPENAI_API_KEY=your_openai_key

# Mistral
MISTRAL_API_KEY=your_mistral_key

# Google Gemini
GEMINI_API_KEY=your_gemini_key

# Kimi (月之暗面)
KIMI_API_KEY=your_kimi_key

# DeepSeek
DEEPSEEK_API_KEY=your_deepseek_key

# Milvus (可选)
MILVUS_HOST=localhost
MILVUS_PORT=19530
MILVUS_URI=your_milvus_uri  # Zilliz Cloud 使用
MILVUS_TOKEN=your_token
```

## CLI 用法

```bash
# 交互式对话
raglight chat

# 带工具的 Agentic 对话
raglight agentic-chat

# 索引文档
raglight index ./my_documents
```

## 架构

```
src/raglight/
├── cli/                    # 命令行界面
├── config/                 # 配置管理
├── cross_encoder/          # 重排序模型
├── document_processing/    # 文档解析
├── embeddings/             # 嵌入模型
├── evaluation/             # RAG 评估 (RAGAS)
├── llm/                    # LLM 实现
├── models/                 # 数据模型
├── rag/                    # 核心 RAG 实现
├── scrapper/               # 网页抓取
└── vectorstore/            # 向量数据库 (Chroma, Milvus)
```

## 支持的文档类型

- PDF
- TXT / Markdown
- HTML
- 代码文件 (Python, JavaScript, TypeScript, Java, C++, C# 等)

## 许可证

MIT 许可证 - 详见 [LICENSE](LICENSE) 文件。

## 贡献

欢迎贡献！请随时提交 Pull Request。

## 致谢

- 文档处理基于 [LangChain](https://github.com/langchain-ai/langchain)
- 向量存储由 [ChromaDB](https://github.com/chroma-core/chroma) 提供支持
- 嵌入模型使用 [Sentence Transformers](https://github.com/UKPLab/sentence-transformers)
