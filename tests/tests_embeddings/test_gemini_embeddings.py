import unittest
from unittest.mock import MagicMock, patch
from raglight.embeddings.gemini_embeddings import GeminiEmbeddingsModel
from ..test_config import TestsConfig


class TestGeminiEmbeddings(unittest.TestCase):
    
    @patch("raglight.embeddings.gemini_embeddings.Client")
    def test_model_load(self, mock_genai: MagicMock):
        """Test that API configuration is called correctly."""
        model = GeminiEmbeddingsModel(TestsConfig.GEMINI_EMBEDDING_MODEL)
        
        mock_genai.configure.assert_called_once()
        self.assertIsNotNone(model.model, "Model (genai module) should be loaded.")

    @patch("raglight.embeddings.gemini_embeddings.Client")
    def test_embed_documents(self, mock_genai: MagicMock):
        """Test document embedding with the correct task_type."""
        mock_genai.embed_content.return_value = {'embedding': [[0.1, 0.2], [0.3, 0.4]]}
        
        model = GeminiEmbeddingsModel(TestsConfig.GEMINI_EMBEDDING_MODEL)
        texts = ["doc1", "doc2"]
        result = model.embed_documents(texts)
        
        self.assertEqual(len(result), 2)
        mock_genai.embed_content.assert_called_with(
            model=TestsConfig.GEMINI_EMBEDDING_MODEL,
            content=texts,
            task_type="retrieval_document"
        )

    @patch("raglight.embeddings.gemini_embeddings.Client")
    def test_embed_query(self, mock_genai: MagicMock):
        """Test query embedding with the correct task_type."""
        mock_genai.embed_content.return_value = {'embedding': [0.1, 0.2]}
        
        model = GeminiEmbeddingsModel(TestsConfig.GEMINI_EMBEDDING_MODEL)
        text = "query"
        result = model.embed_query(text)
        
        self.assertEqual(len(result), 2)
        mock_genai.embed_content.assert_called_with(
            model=TestsConfig.GEMINI_EMBEDDING_MODEL,
            content=text,
            task_type="retrieval_query"
        )


if __name__ == "__main__":
    unittest.main()