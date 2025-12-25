class TestsConfig:
    HUGGINGFACE_EMBEDDINGS = "all-MiniLM-L6-v1"
    OLLAMA_MODEL = "llama3"
    OLLAMA_EMBEDDING_MODEL = "nomic-embed-text"
    GEMINI_LLM_MODEL = "gemini-2.5-flash"
    GEMINI_EMBEDDING_MODEL = "models/gemini-embedding-001"
    CHROMA_PERSIST_DIRECTORY = "tests/tmp/chromadb"
    CHROMA_PERSIST_DIRECTORY_INGESTION = "tests/tmp/chromadb"
    TEST_SYSTEM_PROMPT = "tests/prompts/prompt.txt"
    COLLECTION_NAME = "test"
    DATA_PATH = "tests/tests_vector_store/fixtures"

    OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
