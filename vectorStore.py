from abc import ABC, abstractmethod

class VectorStore(ABC):
    def __init__(self, dataPath):
        self.vectoreStore = None
        self.dataPath = dataPath

    @abstractmethod
    def add(self, name, vector):
        pass

    @abstractmethod
    def load(self):
        pass

    @abstractmethod
    def ingest(self):
        pass
    