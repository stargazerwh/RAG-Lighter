import unittest
import os
from unittest.mock import MagicMock, patch

from raglight.llm.gemini_model import GeminiModel
from ..test_config import TestsConfig


class TestGeminiModel(unittest.TestCase):
    _MOCK_RESPONSE = "Hello! This is a test response."

    @patch("raglight.Settings.GEMINI_API_KEY", "DUMMY_KEY")
    def setUp(self):
        model_name = TestsConfig.GEMINI_LLM_MODEL
        self.model = GeminiModel(
            model_name=model_name,
        )

        mock_response = MagicMock()
        mock_response.text = self._MOCK_RESPONSE
        mock_response.candidates = [MagicMock()]  # Non-empty to pass the check
        mock_generate = MagicMock(return_value=mock_response)

        mock_client = MagicMock()
        mock_client.models.generate_content = mock_generate
        self.model.model = mock_client

    def test_generate_response(self):
        prompt = "Say hello."
        response = self.model.generate({"question": prompt})
        self.assertIsInstance(response, str, "Response should be a string.")
        self.assertGreater(len(response), 0, "Response should not be empty.")
        self.assertEqual(response, self._MOCK_RESPONSE)


if __name__ == "__main__":
    unittest.main()
