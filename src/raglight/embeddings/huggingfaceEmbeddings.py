from typing_extensions import override
from .embeddingsModel import EmbeddingsModel
from langchain_huggingface import HuggingFaceEmbeddings

class HuggingfaceEmbeddings(EmbeddingsModel):

    def __init__(self, model_name):
        super().__init__(model_name)

    @override
    def load(self):
        return HuggingFaceEmbeddings(model_name=self.model_name)
