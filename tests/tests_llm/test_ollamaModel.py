import unittest
from ...src.raglight.llm.ollamaModel import OllamaModel
from ..test_config import TestsConfig


class TestOllamaModel(unittest.TestCase):
    def setUp(self):
        model_name = TestsConfig.OLLAMA_MODEL
        system_prompt_directory = TestsConfig.SYSTEM_PROMPT
        self.model = OllamaModel(
            model_name=model_name, system_prompt_file=system_prompt_directory
        )

    def test_generate_response(self):
        prompt = "Define machine learning."
        response = self.model.generate({"question": prompt})
        self.assertIsInstance(response, str, "Response should be a string.")
        self.assertGreater(len(response), 0, "Response should not be empty.")


if __name__ == "__main__":
    unittest.main()
