from typing_extensions import override
from .llm import LLM
from ollama import Client
from os import environ
from dotenv import load_dotenv
from json import dumps

load_dotenv()

class OllamaModel(LLM):
    def __init__(self, model_name, system_prompt_file = None, role='user'):
        super().__init__(model_name)
        self.role = role
        self.system_prompt = "" if system_prompt_file is None else self.load_system_prompt(system_prompt_file)

    @override
    def load(self):
        return Client(
            host=environ.get('OLLAMA_HOST'),
            headers={'x-some-header': 'some-value'}
            )
    
    @override
    def generate(self, input):
        input["system prompt"] = self.system_prompt
        new_input = dumps(input)
        response = self.model.chat(model=self.model_name, messages=[
            {
                'role': self.role,
                'content': new_input,
            },
        ],
        )
        return response.message.content
    
    @staticmethod
    def load_system_prompt(filePath):
        with open(filePath, 'r', encoding='utf-8') as file:
            prompt = file.read()
        return prompt