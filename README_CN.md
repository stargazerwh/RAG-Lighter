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
- **本地与云端** - 本地 Ollama 或云端 API

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
from raglight.rag import RAG
from raglight.llm import OllamaModel
from raglight.embeddings import HuggingfaceEmbeddingsModel

# 初始化组件
llm = OllamaModel(model_name="llama3")
embeddings = HuggingfaceEmbeddingsModel()

# 创建 RAG 管道
rag = RAG(
    llm=llm,
    embeddings=embeddings,
    persist_directory="./my_db"
)

# 索引文档
rag.index([
    "./documents/file1.pdf",
    "./documents/file2.txt"
])

# 查询
response = rag.query("这些文档的主要内容是什么？")
print(response)
```

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
