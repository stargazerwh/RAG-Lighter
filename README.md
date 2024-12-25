# RAGLight

RAGLight is a lightweight and modular framework for implementing Retrieval-Augmented Generation (RAG). It enhances the capabilities of Large Language Models (LLMs) by combining document retrieval with natural language inference.
Designed for simplicity and flexibility, RAGLight leverages Ollama for LLM interaction and vector embeddings for efficient document similarity searches, making it an ideal tool for building context-aware AI solutions.

---

## Features

- **Embeddings Model**: Uses `all-MiniLM-L6-v2` for creating vector embeddings.
- **LLM**: Employs `llama3` for inference.
- **RAG Pipeline**: Integrates document retrieval and natural language generation into a single workflow.
- **PDF Support**: Currently supports PDF files for ingestion.

---

## Prerequisites

- **Python**: Version >= 3.11
- **Ollama Client**: A properly configured Ollama client.
- **Python Dependencies**: See the Installation section below.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Bessouat40/rag-example.git
cd rag-example
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

```bash
mv .env.example .env
```

Then fill in the .env file with the necessary configuration :

```bash
# Example configuration
OLLAMA_CLIENT=<URL or key for the Ollama client>
PERSIST_DIRECTORY=<Path to store inference data>
PERSIST_DIRECTORY_INGESTION=<Path to store ingestion data>
MODEL_EMBEDDINGS=all-MiniLM-L6-v2
MODEL_NAME=llama3
SYSTEM_PROMPT_DIRECTORY=<Path to the system prompt file>
COLLECTION_NAME=<Collection name for inference>
COLLECTION_NAME_INGESTION=<Collection name for ingestion>
DATA_PATH=./data
```

## Document Ingestion

To ingest your files (currently only PDF files are supported), place them in the `data` folder or the path specified by the `DATA_PATH` variable in the `.env` file.

Run the following script to index the documents :

```bash
python ingestion_example.py
```

This script:

- Loads the embeddings model specified in `.env`.
- Uses the `VectorStore` (Chroma) to index the documents.
- Creates a persistent index in the directory defined by `PERSIST_DIRECTORY_INGESTION`.

## Query the Model (RAG Pipeline)

To query the RAG pipeline, use the following script:

```bash
python rag_example.py
```

The pipeline:

- Retrieves the most relevant documents using the vector model.
- Uses the llama3 model to generate a response based on the retrieved context.

## TODO

- [ ] Feature : Add possibility to use custom pipeline while ingesting data inside Vector Store,
- [ ] Feature : Add new Vector Stores,
- [ ] Feature : Add new LLM Providers (VLLM ?, Huggingface, ...)
