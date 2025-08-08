import unittest
from ...src.raglight.llm.gemini_model import GeminiModel
from ..test_config import TestsConfig

class TestGeminiModel(unittest.TestCase):
    def setUp(self):
        model_name = TestsConfig.GEMINI_LLM_MODEL
        self.model = GeminiModel(
            model_name=model_name
        )

    def test_generate_response(self):
        prompt = "Say hello."
        response = self.model.generate({"question": prompt})
        self.assertIsInstance(response, str, "Response should be a string.")
        self.assertGreater(len(response), 0, "Response should not be empty.")

if __name__ == "__main__":
    unittest.main()
