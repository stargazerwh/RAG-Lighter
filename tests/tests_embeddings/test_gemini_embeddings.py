import unittest
from unittest.mock import MagicMock, patch
from raglight.embeddings.gemini_embeddings import GeminiEmbeddingsModel
from ..test_config import TestsConfig


class TestGeminiEmbeddings(unittest.TestCase):

    @patch("raglight.embeddings.gemini_embeddings.GoogleGenerativeAIEmbeddings")
    def test_model_load(self, mock_embedding: MagicMock):
        mock_embedding.return_value = MagicMock()
        embeddings = GeminiEmbeddingsModel(TestsConfig.GEMINI_EMBEDDING_MODEL)
        self.assertIsNotNone(embeddings.model, "Model should be loaded successfully.")
        mock_embedding.assert_called_once_with(
            model=TestsConfig.GEMINI_EMBEDDING_MODEL, google_api_key=""
        )


if __name__ == "__main__":
    unittest.main()
