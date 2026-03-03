# RAG-Lighter Wiki

> 轻量级模块化 RAG (检索增强生成) 框架完整文档

---

## 📑 目录

1. [项目简介](#项目简介)
2. [核心架构](#核心架构)
3. [快速开始](#快速开始)
4. [组件详解](#组件详解)
5. [高级用法](#高级用法)
6. [API 参考](#api-参考)
7. [配置指南](#配置指南)
8. [最佳实践](#最佳实践)
9. [常见问题](#常见问题)

---

## 项目简介

**RAG-Lighter** 是一个用 Python 编写的轻量级、模块化 RAG (Retrieval-Augmented Generation) 框架。它提供了构建生产级 RAG 应用所需的所有组件，同时保持代码简洁易懂。

### 主要特性

| 特性 | 描述 |
|------|------|
| 🔧 **模块化设计** | 每个组件可独立使用或替换 |
| 🤖 **多 LLM 支持** | OpenAI、Ollama、Mistral、Gemini、Kimi、DeepSeek 等 |
| 💾 **多向量数据库** | ChromaDB、Milvus |
| 🧠 **Agentic RAG** | 支持工具调用的智能检索 |
| 🖼️ **多模态支持** | 文本、代码、图像处理 |
| 📊 **RAG 评估** | 内置 RAGAS 评估指标 |
| 📝 **训练数据生成** | 自动生成问答对用于微调 |
| 🏠 **本地 & 云端** | 本地 Ollama 或云端 API 任选 |

---

## 核心架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        RAG-Lighter 架构                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   数据层 (Data Layer)                                            │
│   ├── Document Processing (PDF, TXT, HTML, Code)                │
│   ├── Web Scraping                                              │
│   └── Data Sources (Folder, GitHub, URL)                        │
│                          ↓                                      │
│   嵌入层 (Embedding Layer)                                       │
│   ├── HuggingFace (Sentence Transformers)                       │
│   ├── OpenAI (text-embedding-ada-002)                           │
│   ├── BGE-M3 (多语言)                                            │
│   ├── Google Gemini                                             │
│   └── Ollama (本地)                                              │
│                          ↓                                      │
│   存储层 (Storage Layer)                                         │
│   ├── ChromaDB (默认)                                            │
│   └── Milvus (分布式)                                            │
│                          ↓                                      │
│   检索层 (Retrieval Layer)                                       │
│   ├── Vector Search                                             │
│   ├── Cross-Encoder Reranking                                   │
│   └── Hybrid Search                                             │
│                          ↓                                      │
│   生成层 (Generation Layer)                                      │
│   ├── OpenAI (GPT-4, GPT-3.5)                                   │
│   ├── Ollama (本地模型)                                          │
│   ├── DeepSeek (支持推理)                                        │
│   ├── Kimi (月之暗面)                                            │
│   ├── Gemini (Google)                                           │
│   └── Mistral / LM Studio                                       │
│                          ↓                                      │
│   评估层 (Evaluation Layer)                                      │
│   └── RAGAS Metrics (Faithfulness, Relevancy, Recall)           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 目录结构

```
src/raglight/
├── cli/                      # 命令行接口
│   ├── main.py
│   └── nltk_management.py
├── config/                   # 配置管理
│   ├── settings.py          # 全局设置
│   ├── rag_config.py        # RAG 配置
│   ├── vector_store_config.py
│   └── agentic_rag_config.py
├── cross_encoder/            # 重排序模型
│   ├── cross_encoder_model.py
│   └── huggingface_cross_encoder.py
├── document_processing/      # 文档解析
│   └── document_loader.py
├── embeddings/               # 嵌入模型
│   ├── embeddings_model.py
│   ├── huggingface_embeddings.py
│   ├── openai_embeddings.py
│   ├── bge_m3_embeddings.py
│   ├── gemini_embeddings.py
│   └── ollama_embeddings.py
├── llm/                      # 大语言模型
│   ├── llm.py               # 基类
│   ├── openai_model.py
│   ├── ollama_model.py
│   ├── deepseek_model.py
│   ├── kimi_model.py
│   ├── gemini_model.py
│   ├── mistral_model.py
│   └── lmstudio_model.py
├── models/                   # 数据模型
│   └── data_source_model.py
├── rag/                      # 核心 RAG 实现
│   ├── rag.py               # 基础 RAG 类
│   ├── builder.py           # 构建器模式
│   ├── simple_rag_api.py    # 简化 API
│   └── rat.py               # RAT (推理增强)
├── scrapper/                 # 网页抓取
├── vectorstore/              # 向量数据库
│   ├── vector_store.py
│   ├── chroma.py
│   └── milvus.py
└── evaluation/               # 评估模块
    └── ragas_evaluator.py
```

---

## 快速开始

### 1. 安装

```bash
# 从 PyPI 安装
pip install raglight

# 或从源码安装
git clone https://github.com/stargazerwh/RAG-Lighter.git
cd RAG-Lighter
pip install -e .
```

### 2. 环境配置

创建 `.env` 文件：

```bash
# OpenAI (可选)
OPENAI_API_KEY=your_openai_key

# Mistral (可选)
MISTRAL_API_KEY=your_mistral_key

# Google Gemini (可选)
GEMINI_API_KEY=your_gemini_key

# Kimi 月之暗面 (可选)
KIMI_API_KEY=your_kimi_key

# DeepSeek (可选)
DEEPSEEK_API_KEY=your_deepseek_key

# Milvus (可选)
MILVUS_HOST=localhost
MILVUS_PORT=19530
```

### 3. 基础使用

#### 方式一：使用 Builder 模式（推荐）

```python
from raglight.rag.builder import Builder
from raglight.config.settings import Settings
from dotenv import load_dotenv

load_dotenv()
Settings.setup_logging()

# 使用 Builder 构建 RAG 管道
rag = (
    Builder()
    .with_embeddings(Settings.HUGGINGFACE, model_name="all-MiniLM-L6-v2")
    .with_vector_store(
        Settings.CHROMA,
        persist_directory="./my_db",
        collection_name="my_collection"
    )
    .with_llm(
        Settings.OLLAMA,
        model_name="llama3.1:8b",
        options={"num_ctx": 8192}
    )
    .build_rag(k=5)
)

# 索引文档
rag.vector_store.ingest(
    data_path="./documents",
    extensions=[".pdf", ".txt", ".md"]
)

# 查询
response = rag.generate("文档的主要内容是什么？")
print(response)
```

#### 方式二：使用简化 API

```python
from raglight.rag.simple_rag_api import RAGPipeline
from raglight.models.data_source_model import FolderSource
from raglight.config.settings import Settings
from raglight.config.rag_config import RAGConfig
from raglight.config.vector_store_config import VectorStoreConfig

# 配置数据源
knowledge_base = [
    FolderSource(path="./knowledge_base"),
]

# 配置向量存储
vector_store_config = VectorStoreConfig(
    embedding_model="all-MiniLM-L6-v2",
    provider=Settings.HUGGINGFACE,
    database=Settings.CHROMA,
    persist_directory="./db",
    collection_name="docs"
)

# 配置 LLM
config = RAGConfig(
    llm="llama3.1:8b",
    provider=Settings.OLLAMA
)

# 创建管道
pipeline = RAGPipeline(config, vector_store_config)
pipeline.build()

# 查询
response = pipeline.generate("你的问题")
print(response)
```

---

## 组件详解

### 1. LLM (大语言模型)

支持多种 LLM 提供商：

```python
# OpenAI
from raglight.llm import OpenAIModel
llm = OpenAIModel(model_name="gpt-4")

# Ollama (本地)
from raglight.llm import OllamaModel
llm = OllamaModel(model_name="llama3.1:8b")

# DeepSeek (支持推理)
from raglight.llm import DeepSeekModel
llm = DeepSeekModel(model_name="deepseek-chat")

# 带推理过程
r1 = DeepSeekModel(model_name="deepseek-reasoner")
result = r1.generate_with_thinking({"question": "复杂问题"})
print(result["reasoning"])  # 推理链
print(result["answer"])     # 最终答案

# Kimi (月之暗面)
from raglight.llm import KimiModel
llm = KimiModel(model_name="kimi-k2.5")

# Gemini
from raglight.llm import GeminiModel
llm = GeminiModel(model_name="gemini-pro")

# Mistral
from raglight.llm import MistralModel
llm = MistralModel(model_name="mistral-large-latest")

# LM Studio (本地 GUI)
from raglight.llm import LMStudioModel
llm = LMStudioModel(
    model_name="local-model",
    api_base="http://localhost:1234/v1"
)
```

### 2. Embedding Models (嵌入模型)

```python
# HuggingFace (默认)
from raglight.embeddings import HuggingfaceEmbeddingsModel
embeddings = HuggingfaceEmbeddingsModel("all-MiniLM-L6-v2")

# BGE-M3 (多语言，推荐)
from raglight.embeddings import BgeM3EmbeddingsModel
embeddings = BgeM3EmbeddingsModel(
    model_name="BAAI/bge-m3",
    use_fp16=True  # 半精度加速
)

# OpenAI
from raglight.embeddings import OpenAIEmbeddingsModel
embeddings = OpenAIEmbeddingsModel(model_name="text-embedding-ada-002")

# Gemini
from raglight.embeddings import GeminiEmbeddingsModel
embeddings = GeminiEmbeddingsModel()

# Ollama (本地)
from raglight.embeddings import OllamaEmbeddingsModel
embeddings = OllamaEmbeddingsModel(model_name="nomic-embed-text")
```

### 3. Vector Store (向量数据库)

#### ChromaDB (默认)

```python
from raglight.vectorstore import ChromaVS
from raglight.embeddings import HuggingfaceEmbeddingsModel

embeddings = HuggingfaceEmbeddingsModel()

vector_store = ChromaVS(
    collection_name="my_collection",
    embeddings_model=embeddings,
    persist_directory="./chroma_db"
)

# 索引文档
vector_store.ingest("./documents")

# 搜索
docs = vector_store.search("查询内容", k=5)
```

#### Milvus (高性能)

```python
from raglight.vectorstore import MilvusVS

# Milvus Lite (本地)
vector_store = MilvusVS(
    collection_name="my_collection",
    embeddings_model=embeddings,
    persist_directory="./milvus_db",
    index_type="HNSW",      # HNSW / IVF_FLAT / IVF_SQ8
    metric_type="COSINE"    # COSINE / L2 / IP
)

# Milvus Server (远程)
vector_store = MilvusVS(
    collection_name="my_collection",
    embeddings_model=embeddings,
    host="localhost",
    port=19530,
    index_type="HNSW",
    metric_type="COSINE"
)

# Zilliz Cloud
vector_store = MilvusVS(
    collection_name="my_collection",
    embeddings_model=embeddings,
    uri="https://your-cluster.zillizcloud.com",
    token="your_api_token"
)
```

### 4. Cross-Encoder (重排序)

```python
from raglight.cross_encoder import HuggingfaceCrossEncoderModel

# 添加重排序提升检索质量
reranker = HuggingfaceCrossEncoderModel(
    model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
)

rag = (
    Builder()
    .with_embeddings(Settings.HUGGINGFACE, model_name="all-MiniLM-L6-v2")
    .with_cross_encoder(Settings.HUGGINGFACE, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
    .with_vector_store(Settings.CHROMA, persist_directory="./db")
    .with_llm(Settings.OLLAMA, model_name="llama3.1:8b")
    .build_rag(k=10)  # 先检索 10 个
)

# RAG 会自动使用 cross-encoder 重排序，返回最相关的 k 个
```

---

## 高级用法

### 1. Agentic RAG

支持工具调用的智能 RAG：

```python
from raglight.rag.builder import Builder
from raglight.config.settings import Settings
from raglight.agent import Agent

# 构建 RAG
rag = (
    Builder()
    .with_embeddings(Settings.HUGGINGFACE, model_name="all-MiniLM-L6-v2")
    .with_vector_store(Settings.CHROMA, persist_directory="./db")
    .with_llm(Settings.OPENAI, model_name="gpt-4")
    .build_rag()
)

# 创建 Agent
agent = Agent(rag=rag)

# Agentic 对话
response = agent.chat(
    "在我的文档中查找关于机器学习的信息，并总结成 3 个要点"
)
```

### 2. 训练数据生成

从文档自动生成问答对：

```python
from raglight.embeddings.build.traindata import load_corpus, documents_to_qa
from raglight.llm import OpenAIModel

# 加载文档
# 参数说明:
# - chunk_size: 每个文档块的大小（字符数）
# - chunk_overlap: 相邻块之间的重叠字符数
# - extensions: 要处理的文件扩展名列表
docs = load_corpus(
    paths=["train.pdf", "docs/"],
    chunk_size=512,
    chunk_overlap=50,
    extensions=[".pdf", ".txt", ".md"],
    verbose=True
)

# 使用 LLM 生成问答对
# 注意：需要 OPENAI_API_KEY 环境变量
llm = OpenAIModel(model_name="gpt-4")
qa_df = documents_to_qa(
    documents=docs,
    llm=llm,
    num_questions_per_doc=3,  # 每个文档块生成的问题数量
    verbose=True
)

# 保存训练数据
qa_df.to_csv("train_qa.csv", index=False)
print(f"生成了 {len(qa_df)} 个问答对")
```

### 3. RAG 评估

使用 RAGAS 指标评估 RAG 质量：

```python
from raglight.evaluation import RAGASEvaluator, RAGResult

# 初始化评估器
# 需要 OPENAI_API_KEY 环境变量用于评估
evaluator = RAGASEvaluator()

# 准备 RAG 结果
result = RAGResult(
    question="什么是 RAG？",
    answer="RAG 是检索增强生成...",
    contexts=[
        "RAG (Retrieval-Augmented Generation) 是一种...",
        "通过检索外部知识来增强 LLM..."
    ]
)

# 单条评估
scores = evaluator.evaluate(result)
print(f"忠实度 (Faithfulness): {scores.faithfulness:.3f}")
print(f"答案相关性 (Answer Relevancy): {scores.answer_relevancy:.3f}")
print(f"上下文相关性 (Context Relevancy): {scores.context_relevancy:.3f}")
print(f"上下文召回率 (Context Recall): {scores.context_recall:.3f}")

# 批量评估
results = [result1, result2, result3]
report = evaluator.generate_report(results)
print(report)
```

**RAGAS 指标说明：**

| 指标 | 说明 | 范围 |
|------|------|------|
| **Faithfulness** | 答案是否忠实于检索的上下文 | 0-1 |
| **Answer Relevancy** | 答案与问题的相关程度 | 0-1 |
| **Context Relevancy** | 检索的上下文与问题的相关程度 | 0-1 |
| **Context Recall** | 上下文是否包含回答问题所需的信息 | 0-1 |

### 4. 多数据源

```python
from raglight.models.data_source_model import FolderSource, GitHubSource, URLSource

knowledge_base = [
    # 本地文件夹
    FolderSource(
        path="./documents",
        extensions=[".pdf", ".txt", ".md"]
    ),
    
    # GitHub 仓库
    GitHubSource(
        url="https://github.com/user/repo",
        branch="main",
        extensions=[".py", ".md"]
    ),
    
    # 网页
    URLSource(
        url="https://example.com/docs",
        depth=2  # 爬取深度
    )
]

# 在 RAGPipeline 中使用
config = RAGConfig(knowledge_base=knowledge_base)
```

---

## API 参考

### Builder 类

构建 RAG 管道的流式接口。

```python
class Builder:
    def with_embeddings(self, type: str, **kwargs) -> Builder
    def with_cross_encoder(self, type: str, **kwargs) -> Builder
    def with_vector_store(self, type: str, **kwargs) -> Builder
    def with_llm(self, type: str, **kwargs) -> Builder
    def build_rag(self, k: int = 5) -> RAG
    def build_rat(self, k: int = 5) -> RAT  # 推理增强生成
```

### RAG 类

```python
class RAG:
    def generate(self, query: str) -> str
    def generate_with_context(self, query: str, contexts: List[str]) -> str
    def add_documents(self, documents: List[str])
    def clear_cache()
```

### RAGPipeline 类

简化版 API。

```python
class RAGPipeline:
    def __init__(self, config: RAGConfig, vector_store_config: VectorStoreConfig)
    def build()
    def generate(self, query: str) -> str
    def ingest(self, sources: List[DataSource])
```

---

## 配置指南

### Settings 常量

```python
from raglight.config.settings import Settings

# LLM 提供商
Settings.OPENAI
Settings.OLLAMA
Settings.MISTRAL
Settings.GOOGLE_GEMINI
Settings.LMSTUDIO
Settings.KIMI
Settings.DEEPSEEK

# 向量数据库
Settings.CHROMA
Settings.MILVUS

# 嵌入模型提供商
Settings.HUGGINGFACE
Settings.BGE_M3

# 默认值
Settings.DEFAULT_EMBEDDINGS_MODEL  # "all-MiniLM-L6-v2"
Settings.DEFAULT_LLM              # "llama3.1:8b"
Settings.DEFAULT_K                # 5
Settings.DEFAULT_COLLECTION_NAME  # "rag_collection"
```

### RAGConfig

```python
from raglight.config.rag_config import RAGConfig

config = RAGConfig(
    llm="gpt-4",                    # 模型名称
    provider=Settings.OPENAI,       # 提供商
    api_base=None,                  # 自定义 API 地址
    k=5,                            # 检索数量
    cross_encoder_model=None,       # 重排序模型
    system_prompt=None,             # 系统提示词
    knowledge_base=None             # 数据源列表
)
```

### VectorStoreConfig

```python
from raglight.config.vector_store_config import VectorStoreConfig

config = VectorStoreConfig(
    embedding_model="all-MiniLM-L6-v2",
    provider=Settings.HUGGINGFACE,
    api_base=None,
    database=Settings.CHROMA,
    persist_directory="./db",
    collection_name="docs",
    index_type="HNSW",      # Milvus 专用
    metric_type="COSINE"    # Milvus 专用
)
```

---

## 最佳实践

### 1. 选择合适的嵌入模型

| 场景 | 推荐模型 | 维度 |
|------|----------|------|
| 英文通用 | all-MiniLM-L6-v2 | 384 |
| 英文高质量 | all-mpnet-base-v2 | 768 |
| 多语言 | BAAI/bge-m3 | 1024 |
| 中文 | BAAI/bge-large-zh-v1.5 | 1024 |

### 2. 分块策略

```python
# 小文档（< 1000 字符）
chunk_size = 512
chunk_overlap = 50

# 中等文档（1000-10000 字符）
chunk_size = 1024
chunk_overlap = 100

# 大文档（> 10000 字符）
chunk_size = 2048
chunk_overlap = 200
```

### 3. 检索优化

```python
# 基础检索
k = 5

# 使用重排序（推荐）
k = 20  # 先检索更多
top_k = 5  # 重排序后取 Top 5

# 混合检索（ChromaDB 支持）
# 结合向量检索 + 关键词检索
```

### 4. 生产环境建议

1. **使用 Milvus**：ChromaDB 适合原型，Milvus 适合生产
2. **添加重排序**：Cross-Encoder 能显著提升检索质量
3. **缓存嵌入**：避免重复计算
4. **监控评估**：定期使用 RAGAS 评估系统性能
5. **日志记录**：使用 `Settings.setup_logging()`

---

## 常见问题

### Q: 如何切换不同的 LLM？

```python
# 只需修改 Builder 中的 LLM 配置
rag = (
    Builder()
    .with_embeddings(Settings.HUGGINGFACE, model_name="all-MiniLM-L6-v2")
    .with_vector_store(Settings.CHROMA, persist_directory="./db")
    .with_llm(Settings.OPENAI, model_name="gpt-4")  # 切换这里
    .build_rag()
)
```

### Q: 支持哪些文档格式？

- PDF
- TXT / Markdown
- HTML
- 代码文件 (.py, .js, .ts, .java, .cpp, .cs, etc.)

### Q: 如何处理大文档？

```python
# 使用流式处理或增大分块大小
docs = load_corpus(
    paths=["large_doc.pdf"],
    chunk_size=2048,  # 增大分块
    chunk_overlap=200,
    verbose=True
)
```

### Q: 本地部署需要什么配置？

```bash
# 最小配置
# - CPU: 4 核
# - 内存: 8GB
# - 磁盘: 10GB

# 推荐配置（运行 7B 模型）
# - CPU: 8 核
# - 内存: 16GB
# - GPU: 8GB VRAM (可选)

# 安装 Ollama
curl -fsSL https://ollama.com/install.sh | sh

# 拉取模型
ollama pull llama3.1:8b
ollama pull nomic-embed-text
```

### Q: 如何调试检索结果？

```python
# 开启详细日志
import logging
logging.basicConfig(level=logging.DEBUG)

# 查看检索到的文档
retrieved_docs = rag.vector_store.search("查询", k=5)
for i, doc in enumerate(retrieved_docs):
    print(f"[{i}] {doc.page_content[:200]}...")
```

### Q: 支持流式输出吗？

目前版本暂不支持，将在后续版本中添加。

---

## 相关链接

- **GitHub**: https://github.com/stargazerwh/RAG-Lighter
- **PyPI**: https://pypi.org/project/raglight
- **Issues**: https://github.com/stargazerwh/RAG-Lighter/issues

---

*Last updated: 2026-03-03*
