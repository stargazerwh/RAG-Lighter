from ..llm.ollamaModel import OllamaModel
from ..vectorestore.chroma import ChromaVS
from ..config.settings import Settings
from .rag import RAG
from ..embeddings.huggingfaceEmbeddings import HuggingfaceEmbeddings

class Builder:
    def __init__(self):
        self.vector_store = None
        self.embeddings = None
        self.llm = None
        self.rag = None

    def with_embeddings(self, type, **kwargs):
        print("⏳ Creating an Embeddings Model...")
        if type == Settings.HUGGINGFACE:
            self.embeddings = HuggingfaceEmbeddings(**kwargs)
        else:
            raise ValueError(f"Unknown Embeddings Model type: {type}")
        print("✅ Embeddings Model created")
        return self

    def with_vector_store(self, type: str, **kwargs):
        print("⏳ Creating a VectorStore...")
        if self.embeddings is None:
            raise ValueError(f"You need to set an embedding model before setting a vector store")
        elif type == Settings.CHROMA:
            self.vector_store = ChromaVS(embeddings_model=self.embeddings, **kwargs)
        else:
            raise ValueError(f"Unknown VectorStore type: {type}")
        print("✅ VectorStore created")
        return self
    
    def with_llm(self, type, **kwargs):
        print("⏳ Creating an LLM...")
        if type == Settings.OLLAMA:
            self.llm = OllamaModel(**kwargs)
        else:
            raise ValueError(f"Unknown LLM type: {type}")
        print("✅ LLM created")
        return self

    def build_rag(self):
        if self.vector_store is None:
            raise ValueError("VectorStore is required")
        if self.llm is None:
            raise ValueError("LLM is required")
        if self.embeddings is None:
            raise ValueError("Embeddings Model is required")
        self.rag = RAG(self.embeddings, self.vector_store, self.llm)
        return self.rag
    
    def build_vector_store(self):
        if self.vector_store is None:
            raise ValueError("VectorStore is required")
        if self.embeddings is None:
            raise ValueError("Embeddings Model is required")
        return self.vector_store

    def build_llm(self):
        if self.llm is None:
            raise ValueError("LLM is required")
        return self.llm