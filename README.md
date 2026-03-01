# RAG-Lighter

A lightweight, modular Retrieval-Augmented Generation (RAG) framework for Python.

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## Features

- **Modular Design** - Easy to extend and customize
- **Multiple LLM Providers** - Support for various AI models
- **Agentic RAG** - Advanced retrieval with tool usage
- **Multi-modal Support** - Text, code, and image processing
- **Local & Cloud** - Run locally with Ollama or use cloud APIs

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
from raglight.rag import RAG
from raglight.llm import OllamaModel
from raglight.embeddings import HuggingfaceEmbeddingsModel

# Initialize components
llm = OllamaModel(model_name="llama3")
embeddings = HuggingfaceEmbeddingsModel()

# Create RAG pipeline
rag = RAG(
    llm=llm,
    embeddings=embeddings,
    persist_directory="./my_db"
)

# Index documents
rag.index([
    "./documents/file1.pdf",
    "./documents/file2.txt"
])

# Query
response = rag.query("What is the main topic of these documents?")
print(response)
```

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
├── llm/                    # LLM implementations
├── models/                 # Data models
├── rag/                    # Core RAG implementation
├── scrapper/               # Web scraping
└── vectorstore/            # Vector database
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
