"""Simple Ollama client wrapper for local LLM and embedding usage."""

from __future__ import annotations

from typing import Any, Dict, Optional


class OllamaService:
    def __init__(self, model_name: str = "llama3", client: Optional[Any] = None):
        self.model_name = model_name
        self.client = client

    def generate(self, prompt: str, model: Optional[str] = None) -> Dict[str, str]:
        if self.client is None:
            raise ValueError("An Ollama client is required.")

        response = self.client.generate(model=model or self.model_name, prompt=prompt)
        reply = response.get("response", "")
        return {"prompt": prompt, "response": reply, "reply": reply}

    def embeddings(self, model: str, prompt: str) -> Dict[str, Any]:
        if self.client is None:
            raise ValueError("An Ollama client is required.")

        return self.client.embeddings(model=model, prompt=prompt)
