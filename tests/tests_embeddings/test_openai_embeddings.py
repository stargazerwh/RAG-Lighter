import unittest
from unittest.mock import MagicMock, patch
from raglight.embeddings.openai_embeddings import OpenAIEmbeddingsModel
from ..test_config import TestsConfig


class TestOpenAIEmbeddings(unittest.TestCase):
    
    @patch("raglight.embeddings.openai_embeddings.OpenAI")
    def test_model_load(self, mock_openai: MagicMock):
        """Test OpenAI client initialization."""
        mock_openai.return_value = MagicMock()
        model = OpenAIEmbeddingsModel(TestsConfig.OPENAI_EMBEDDING_MODEL)
        
        self.assertIsNotNone(model.model)
        mock_openai.assert_called_once()

    @patch("raglight.embeddings.openai_embeddings.OpenAI")
    def test_embed_documents(self, mock_openai: MagicMock):
        """Test document embedding (batch)."""
        mock_client = mock_openai.return_value
        
        mock_data_1 = MagicMock()
        mock_data_1.embedding = [0.1, 0.1]
        mock_data_2 = MagicMock()
        mock_data_2.embedding = [0.2, 0.2]
        
        mock_response = MagicMock()
        mock_response.data = [mock_data_1, mock_data_2]
        
        mock_client.embeddings.create.return_value = mock_response

        model = OpenAIEmbeddingsModel(TestsConfig.OPENAI_EMBEDDING_MODEL)
        result = model.embed_documents(["text1", "text2"])
        
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], [0.1, 0.1])
        mock_client.embeddings.create.assert_called_with(
            input=["text1", "text2"],
            model=TestsConfig.OPENAI_EMBEDDING_MODEL
        )

    @patch("raglight.embeddings.openai_embeddings.OpenAI")
    def test_embed_query(self, mock_openai: MagicMock):
        """Test single query embedding."""
        mock_client = mock_openai.return_value
        
        mock_data = MagicMock()
        mock_data.embedding = [0.5, 0.5]
        
        mock_response = MagicMock()
        mock_response.data = [mock_data]
        
        mock_client.embeddings.create.return_value = mock_response

        model = OpenAIEmbeddingsModel(TestsConfig.OPENAI_EMBEDDING_MODEL)
        result = model.embed_query("query")
        
        self.assertEqual(result, [0.5, 0.5])
        mock_client.embeddings.create.assert_called_with(
            input=["query"],
            model=TestsConfig.OPENAI_EMBEDDING_MODEL
        )


if __name__ == "__main__":
    unittest.main()