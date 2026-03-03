# RAG-Lighter

A lightweight, modular Retrieval-Augmented Generation (RAG) framework for Python.

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

[中文文档](README_CN.md) | [English](README.md)

## Features

- **Modular Design** - Easy to extend and customize
- **Multiple LLM Providers** - Support for various AI models
- **Multiple Vector Stores** - ChromaDB, Milvus support
- **Agentic RAG** - Advanced retrieval with tool usage
- **Multi-modal Support** - Text, code, and image processing
- **RAG Evaluation** - RAGAS metrics for quality assessment
- **Training Data Generation** - Auto-generate QA pairs from documents
- **Local & Cloud** - Run locally with Ollama or use cloud APIs
- **🔥 Parent-Child Chunking** - Small chunks for retrieval, large chunks for generation
- **🔥 Query Rewriting** - Auto/Direct/HyDE/Subquery strategies with LLM-based selection

## Supported LLM Providers

| Provider | Status | Notes |
|----------|--------|-------|
| **OpenAI** | ✅ Supported | GPT-4, GPT-3.5, etc. |
| **Ollama** | ✅ Supported | Local models |
| **Mistral** | ✅ Supported | Mistral AI API |
| **Google Gemini** | ✅ Supported | Gemini Pro, etc. |
| **LM Studio** | ✅ Supported | Local GUI management |
| **Kimi (Moonshot)** | ✅ Supported | OpenAI-compatible API |
| **DeepSeek** | ✅ Supported | With reasoning support |

## Supported Vector Stores

| Vector Store | Status | Notes |
|--------------|--------|-------|
| **ChromaDB** | ✅ Supported | Default, local/remote |
| **Milvus** | ✅ Supported | With vector indexing (IVF_FLAT, HNSW, etc.) |

## Supported Embedding Models

| Model | Status | Notes |
|-------|--------|-------|
| **Sentence Transformers** | ✅ Supported | all-MiniLM-L6-v2, etc. |
| **BGE-M3** | ✅ Supported | Multilingual, 1024-dim |
| **OpenAI** | ✅ Supported | text-embedding-ada-002 |
| **Google Gemini** | ✅ Supported | gemini-embedding-001 |

## Installation

```bash
pip install raglight
```

Or install from source:

```bash
git clone https://github.com/stargazerwh/RAG-lightweight.git
cd RAG-lightweight
pip install -e .
```

## Quick Start

### Basic Usage

```python
from raglight.rag.builder import Builder
from raglight.config.settings import Settings

# Create RAG pipeline
rag = (
    Builder()
    .with_embeddings(Settings.HUGGINGFACE, model_name="all-MiniLM-L6-v2")
    .with_vector_store(Settings.CHROMA, collection_name="my_docs")
    .with_llm(Settings.OLLAMA, model_name="llama3.1:8b")
    .build_rag(k=5)
)

# Index documents
rag.vector_store.ingest("./documents")

# Query
response = rag.generate("What is the main topic?")
print(response)
```

### 🔥 Parent-Child Chunking (Recommended)

Small chunks for precise retrieval, large chunks for complete context:

```python
from raglight.config.rag_config import RAGConfig

# Configure parent-child chunking
config = RAGConfig(
    llm="llama3.1:8b",
    provider=Settings.OLLAMA,
    use_parent_child_chunking=True,
    parent_chunk_size=2000,    # Large chunks for LLM context
    child_chunk_size=400,      # Small chunks for retrieval
    query_rewrite_strategy="Auto"  # Auto-select strategy
)

rag = (
    Builder()
    .with_embeddings(Settings.HUGGINGFACE, model_name="all-MiniLM-L6-v2")
    .with_vector_store(Settings.CHROMA, collection_name="pc_docs")
    .with_llm(Settings.OLLAMA, model_name="llama3.1:8b")
    .with_cross_encoder(Settings.HUGGINGFACE, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
    .build_rag(config=config)
)

# Cross-encoder reranks child chunks, returns parent chunks
response = rag.generate("Your complex question?")
print(f"Strategy used: {rag.get_last_strategy()}")  # Direct/HyDE/Subquery
```

### 🔥 Query Rewriting Strategies

```python
from raglight.config.rag_config import RAGConfig

config = RAGConfig(
    llm="llama3.1:8b",
    provider=Settings.OLLAMA,
    query_rewrite_strategy="HyDE"  # or "Direct", "Subquery", "Auto"
)

rag = Builder().with_embeddings(...).with_vector_store(...).with_llm(...).build_rag(config=config)

# HyDE: Generates hypothetical document for better retrieval
# Subquery: Decomposes complex questions into sub-queries
# Auto: LLM selects the best strategy dynamically
```

### Query Strategy Comparison

| Strategy | Use Case | Description |
|----------|----------|-------------|
| **Direct** | Simple, clear questions | Query as-is |
| **HyDE** | Abstract/conceptual questions | Generate hypothetical answer document |
| **Subquery** | Complex multi-part questions | Decompose into sub-queries |
| **Auto** | Unknown question type | LLM dynamically selects strategy |

### Using Kimi (Moonshot AI)

```python
from raglight.llm import KimiModel

kimi = KimiModel(model_name="kimi-k2.5")
response = kimi.generate({"question": "Hello, how are you?"})
print(response)
```

### Using DeepSeek

```python
from raglight.llm import DeepSeekModel

# Standard chat model
deepseek = DeepSeekModel(model_name="deepseek-chat")
response = deepseek.generate({"question": "Explain quantum computing"})

# Reasoning model (R1)
r1 = DeepSeekModel(model_name="deepseek-reasoner")
result = r1.generate_with_thinking({"question": "Solve this complex problem..."})
print(result["reasoning"])  # Chain of thought
print(result["answer"])     # Final answer
```

### Using Milvus Vector Store

```python
from raglight.vectorstore import MilvusVS
from raglight.embeddings import HuggingfaceEmbeddingsModel

# Initialize Milvus with vector indexing
embeddings = HuggingfaceEmbeddingsModel("all-MiniLM-L6-v2")

# Local Milvus
vector_store = MilvusVS(
    collection_name="my_collection",
    embeddings_model=embeddings,
    persist_directory="./milvus_db",  # Milvus Lite mode
    index_type="HNSW",  # or "IVF_FLAT", "IVF_SQ8"
    metric_type="COSINE"  # or "L2", "IP"
)

# Or connect to Milvus server
vector_store = MilvusVS(
    collection_name="my_collection",
    embeddings_model=embeddings,
    host="localhost",
    port=19530,
    index_type="HNSW",
    metric_type="COSINE"
)
```

### Using BGE-M3 Embeddings

```python
from raglight.embeddings import BgeM3EmbeddingsModel

# BGE-M3 for multilingual retrieval
embeddings = BgeM3EmbeddingsModel(
    model_name="BAAI/bge-m3",
    use_fp16=True  # Use half precision for faster inference
)

# Or use BGE-Large
embeddings = BgeM3EmbeddingsModel("BAAI/bge-large-en-v1.5")
```

### RAG Evaluation with RAGAS

```python
from raglight.evaluation import RAGASEvaluator, RAGResult

# Initialize evaluator
evaluator = RAGASEvaluator()

# Create RAG result
result = RAGResult(
    question="What is the main topic?",
    answer="The document discusses...",
    contexts=retrieved_documents
)

# Evaluate
scores = evaluator.evaluate(result)
print(f"Faithfulness: {scores.faithfulness}")
print(f"Answer Relevancy: {scores.answer_relevancy}")
print(f"Context Relevancy: {scores.context_relevancy}")
print(f"Context Recall: {scores.context_recall}")

# Batch evaluation
results = [result1, result2, result3]
report = evaluator.generate_report(results)
```

### Training Data Generation

Automatically generate question-answer pairs from your documents for fine-tuning or evaluation:

```python
from raglight.embeddings.build.traindata import load_corpus, documents_to_qa
from raglight.llm import OpenAIModel

# Load and chunk documents
docs = load_corpus(["train.pdf"], chunk_size=512, verbose=True)

# Generate QA pairs using LLM
llm = OpenAIModel(model_name="gpt-4")
qa_df = documents_to_qa(docs, llm=llm, verbose=True)

# Save for training
qa_df.to_csv("train_qa.csv", index=False)
```

### Agentic RAG

```python
from raglight.agent import Agent

agent = Agent(rag=rag)
response = agent.chat("Find information about X in my documents")
```

## Environment Variables

Create a `.env` file:

```bash
# OpenAI
OPENAI_API_KEY=your_openai_key

# Mistral
MISTRAL_API_KEY=your_mistral_key

# Google Gemini
GEMINI_API_KEY=your_gemini_key

# Kimi (Moonshot)
KIMI_API_KEY=your_kimi_key

# DeepSeek
DEEPSEEK_API_KEY=your_deepseek_key

# Milvus (optional)
MILVUS_HOST=localhost
MILVUS_PORT=19530
MILVUS_URI=your_milvus_uri  # For Zilliz Cloud
MILVUS_TOKEN=your_token
```

## CLI Usage

```bash
# Interactive chat
raglight chat

# Agentic chat with tools
raglight agentic-chat

# Index documents
raglight index ./my_documents
```

## Architecture

```
src/raglight/
├── cli/                    # Command line interface
├── config/                 # Configuration management
├── cross_encoder/          # Re-ranking models
├── document_processing/    # Document parsing
├── embeddings/             # Embedding models
├── evaluation/             # RAG evaluation (RAGAS)
├── llm/                    # LLM implementations
├── models/                 # Data models
├── rag/                    # Core RAG implementation
├── scrapper/               # Web scraping
└── vectorstore/            # Vector database (Chroma, Milvus)
```

## Supported Document Types

- PDF
- TXT / Markdown
- HTML
- Code files (Python, JavaScript, TypeScript, Java, C++, C#, etc.)

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- Built with [LangChain](https://github.com/langchain-ai/langchain) for document processing
- Vector storage powered by [ChromaDB](https://github.com/chroma-core/chroma)
- Embeddings via [Sentence Transformers](https://github.com/UKPLab/sentence-transformers)
