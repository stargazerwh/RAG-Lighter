from abc import ABC, abstractmethod

class VectorStore(ABC):
    def __init__(self, persist_directory, embeddings_model):
        self.embeddings_model = embeddings_model.get_model()
        self.persist_directory = persist_directory
        self.vectoreStore = None

    @abstractmethod
    def ingest(self, **kwargs):
        pass
    
    @abstractmethod
    def similarity_search(self, question, k=2):
        pass