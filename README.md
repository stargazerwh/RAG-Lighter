# RAG Example

## Introduction

This repository demonstrates the use of Retrieval-Augmented Generation (RAG) to enhance the capabilities of a Large Language Model (LLM) by combining document retrieval with language inference. The application uses Ollama to serve the LLM and leverages vector embeddings for efficient document similarity searches.

## Features

- Uses `all-MiniLM-L6-v2` for embeddings creation.
- Employs `llama3` for LLM inference.
- Integrates document retrieval with natural language generation.

### Prerequisites

- Python >= 3.11
- An Ollama client properly configured

## Installation

### Requirements

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/Bessouat40/rag-example.git
cd rag-example
pip install -r requirements.txt
```

### Environment variables

```bash
mv .env.example .env
```

Then fill your env file with your ollama client.

## Test code

```bash
python main.py
```
