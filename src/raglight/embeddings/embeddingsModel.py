from abc import ABC, abstractmethod

class EmbeddingsModel(ABC):

    def __init__(self, model_name):
        self.model_name = model_name
        self.model = self.load()

    @abstractmethod
    def load(self):
        pass

    def get_model(self):
        return self.model