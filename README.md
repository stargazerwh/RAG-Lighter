# RAGLight

![License](https://img.shields.io/github/license/Bessouat40/RAGLight)
[![Downloads](https://static.pepy.tech/personalized-badge/raglight?period=total&units=international_system&left_color=grey&right_color=red&left_text=Downloads)](https://pepy.tech/projects/raglight)

<div align="center">
    <img alt="ollama" height="200px" src="./media/raglight.png">
</div>

**RAGLight** is a lightweight and modular Python library for implementing **Retrieval-Augmented Generation (RAG)**. It enhances the capabilities of Large Language Models (LLMs) by combining document retrieval with natural language inference.

Designed for simplicity and flexibility, RAGLight provides modular components to easily integrate various LLMs, embeddings, and vector stores, making it an ideal tool for building context-aware AI solutions. ✨

---

> ## Requirements
>
> Actually you need Ollama running on your computer.

## Features 🔥

- 🌐 **Embeddings Model Integration**: Plug in your preferred embedding models (e.g., HuggingFace `all-MiniLM-L6-v2`) for compact and efficient vector embeddings.
- 🧙🏽 **LLM Agnostic**: Seamlessly integrates with different LLMs, such as `llama3` or custom providers, for natural language inference.
- ⚖️ **RAG Pipeline**: Combines document retrieval and language generation in a unified workflow.
- 🖋️ **Flexible Document Support**: Ingest and index various document types (e.g., PDF, TXT, DOCX).
- 🛠️ **Extensible Architecture**: Easily swap vector stores, embedding models, or LLMs to suit your needs.

---

## Import library 🛠️

If you want to install library, use :

```bash
pip install raglight
```

## Installation without import 🛠️

- Install RAGLight directly from PyPI:

```bash
poetry install
```

Then you can run your code :

```bash
poetry run python my_code.py
```

- If you want to use `pip install` :

```bash
pip install -r requirements.txt
```

---

## Quick Start 🚀

### Use Default Pipeline ⭐️

You can setup with two lines your RAG with RAGLight:

```python
from raglight.rag.simple_rag_api import RAGPipeline
from raglight.models.data_source_model import FolderSource, GitHubSource
from raglight.config.settings import Settings

Settings.setup_logging()

pipeline = RAGPipeline(knowledge_base=[
FolderSource(path="<path to your folder with pdf>/knowledge_base"),
GitHubSource(url="https://github.com/Bessouat40/RAGLight")
], model_name="llama3")

pipeline.build()

response = pipeline.generate("How can I create an easy RAGPipeline using raglight framework ? Give me python implementation")
print(response)
```

### **1. Configure Your Pipeline**

Set up the components of your RAG pipeline:

```python
from raglight.rag.builder import Builder
from raglight.config.settings import Settings

rag = Builder() \
    .with_embeddings(Settings.HUGGINGFACE, model_name=model_embeddings) \
    .with_vector_store(Settings.CHROMA, persist_directory=persist_directory, collection_name=collection_name) \
    .with_llm(Settings.OLLAMA, model_name=model_name, system_prompt_file=system_prompt_directory) \
    .build_rag()
```

### **2. Ingest Documents**

Use the pipeline to ingest documents into the vector store:

```python
rag.vector_store.ingest(file_extension='**/*.pdf', data_path='./data')
```

### **3. Query the Pipeline**

Retrieve and generate answers using the RAG pipeline:

```python
response = rag.question_graph("How can I optimize my marathon training?")
print(response)
```

---

## Advanced Configuration ⚙️

### Environment Variables

Configure the pipeline with environment variables for better modularity:

```bash
export PERSIST_DIRECTORY=./vectorstore
export MODEL_EMBEDDINGS=all-MiniLM-L6-v2
export MODEL_NAME=llama3
```

You can also define these in a `.env` file:

```bash
PERSIST_DIRECTORY=./vectorstore
MODEL_EMBEDDINGS=all-MiniLM-L6-v2
MODEL_NAME=llama3
```

---

## TODO

- [ ] **Feature**: Add the possibility to use custom pipelines while ingesting data into the Vector Store.
- [ ] **Feature**: Add support for new Vector Stores (e.g., FAISS, Weaviate, Milvus).
- [ ] **Feature**: Integrate new LLM providers (e.g., VLLM, HuggingFace, GPT-Neo).

---

🚀 **Get started with RAGLight today and build smarter, context-aware AI solutions!**
