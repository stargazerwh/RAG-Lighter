from abc import ABC, abstractmethod

class LLM(ABC):

    def __init__(self, model_name):
        self.model_name = model_name
        self.model = self.load()

    @abstractmethod
    def load(self):
        pass
    
    @abstractmethod
    def generate(self, input: dict):
        pass