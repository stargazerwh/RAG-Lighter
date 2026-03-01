# RAG-Lighter

轻量级模块化 RAG (检索增强生成) 框架

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## 特性

- **模块化设计** - 易于扩展和定制
- **多 LLM 提供商** - 支持多种 AI 模型
- **Agentic RAG** - 支持工具调用的智能检索
- **多模态支持** - 文本、代码和图像处理
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
├── llm/                    # LLM 实现
├── models/                 # 数据模型
├── rag/                    # 核心 RAG 实现
├── scrapper/               # 网页抓取
└── vectorstore/            # 向量数据库
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
