import unittest
from unittest.mock import MagicMock, patch
from raglight.embeddings.huggingface_embeddings import HuggingfaceEmbeddingsModel
from ..test_config import TestsConfig


class TestHuggingFaceEmbeddings(unittest.TestCase):

    @patch("raglight.embeddings.huggingface_embeddings.HuggingFaceEmbeddings")
    def test_model_load(self, mock_embedding: MagicMock):
        mock_embedding.return_value = MagicMock
        embeddings = HuggingfaceEmbeddingsModel(TestsConfig.HUGGINGFACE_EMBEDDINGS)
        self.assertIsNotNone(embeddings.model, "Model should be loaded successfully.")
        mock_embedding.assert_called_once_with(
            model_name=TestsConfig.HUGGINGFACE_EMBEDDINGS
        )


if __name__ == "__main__":
    unittest.main()
