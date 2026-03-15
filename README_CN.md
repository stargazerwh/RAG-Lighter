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
git clone https://github.com/stargazerwh/RAG-Lighter.git
cd RAG-Lighter
pip install -e .
```

## 快速开始
