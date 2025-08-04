from __future__ import annotations
from typing import Optional, Dict, Any
from typing_extensions import override
from ..config.settings import Settings
from .llm import LLM
from json import dumps
import logging

from openai import OpenAI


class LMStudioModel(LLM):
    def __init__(
        self,
        model_name: str,
        system_prompt: Optional[str] = None,
        system_prompt_file: Optional[str] = None,
        api_base: Optional[str] = None,
        role: str = "user",
    ) -> None:
        self.api_base = api_base or Settings.DEFAULT_LMSTUDIO_CLIENT
        super().__init__(model_name, system_prompt, system_prompt_file, self.api_base)
        logging.info(f"Using LMStudio with {model_name} model 🤖")
        self.role: str = role

    @override
    def load(self) -> OpenAI:
        return OpenAI(base_url=self.api_base, api_key="lm-studio")

    @override
    def generate(self, input: Dict[str, Any]) -> str:
        new_input = dumps(input)
        response = self.model.chat.completions.create(
            model="model-identifier",
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": new_input},
            ],
            temperature=0.7,
        )
        return response.choices[0].message.content
