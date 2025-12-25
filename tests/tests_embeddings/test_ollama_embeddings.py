import unittest
from unittest.mock import MagicMock, patch
from raglight.embeddings.ollama_embeddings_model import OllamaEmbeddingsModel
from ..test_config import TestsConfig


class TestOllamaEmbeddings(unittest.TestCase):
    
    @patch("raglight.embeddings.ollama_embeddings_model.Client")
    def test_model_load(self, mock_client: MagicMock):
        """Test Ollama client initialization."""
        mock_client.return_value = MagicMock()
        embeddings = OllamaEmbeddingsModel(TestsConfig.OLLAMA_EMBEDDING_MODEL)
        self.assertIsNotNone(embeddings.model)
        mock_client.assert_called_once()

    @patch("raglight.embeddings.ollama_embeddings_model.Client")
    def test_embed_documents(self, mock_client: MagicMock):
        """Test batch embedding with .embed() method."""
        mock_instance = mock_client.return_value
        mock_instance.embed.return_value = {
            "embeddings": [[0.1, 0.2], [0.3, 0.4]]
        }
        
        model = OllamaEmbeddingsModel(TestsConfig.OLLAMA_EMBEDDING_MODEL)
        texts = ["doc1", "doc2"]
        result = model.embed_documents(texts)
        
        self.assertEqual(len(result), 2)
        mock_instance.embed.assert_called_with(
            model=TestsConfig.OLLAMA_EMBEDDING_MODEL,
            input=texts,
            options=model.options
        )

    @patch("raglight.embeddings.ollama_embeddings_model.Client")
    def test_embed_query(self, mock_client: MagicMock):
        """Test single embedding with .embeddings() method."""
        mock_instance = mock_client.return_value
        mock_instance.embeddings.return_value = {
            "embedding": [0.9, 0.9]
        }
        
        model = OllamaEmbeddingsModel(TestsConfig.OLLAMA_EMBEDDING_MODEL)
        text = "query text"
        result = model.embed_query(text)
        
        self.assertEqual(result, [0.9, 0.9])
        mock_instance.embeddings.assert_called_with(
            model=TestsConfig.OLLAMA_EMBEDDING_MODEL,
            prompt=text,
            options=model.options
        )


if __name__ == "__main__":
    unittest.main()