from __future__ import annotations

from typing import List, Dict
import requests


class OllamaClient:
    """
    Free local LLM via Ollama (https://ollama.com).
    No API key needed — runs entirely on your machine.

    Setup:
        1. Install Ollama: https://ollama.com/download
        2. Pull a model:  ollama pull llama3.2
        3. Ollama serves at http://localhost:11434 automatically.
    """

    def __init__(self, host: str = "http://localhost:11434", model: str = "llama3.2"):
        self.host = host.rstrip("/")
        self.default_model = model

    def generate(self, prompt: str, model: str = None) -> str:
        """Single-turn generation (used for column intelligence)."""
        model = model or self.default_model
        response = requests.post(
            f"{self.host}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "format": "json",
            },
            timeout=60,
        )
        response.raise_for_status()
        return response.json()["response"]

    def chat_completion(self, messages: List[Dict], model: str = None) -> str:
        """Multi-turn chat with system/user/assistant messages (used for chatbot)."""
        model = model or self.default_model
        response = requests.post(
            f"{self.host}/api/chat",
            json={
                "model": model,
                "messages": messages,
                "stream": False,
            },
            timeout=120,
        )
        response.raise_for_status()
        return response.json()["message"]["content"]


class GroqClient:
    """
    Free cloud LLM via Groq (https://console.groq.com).
    Get a free API key at console.groq.com — no credit card needed.

    Free-tier models (as of 2025):
        llama-3.3-70b-versatile   — best quality
        llama-3.1-8b-instant      — fastest
        mixtral-8x7b-32768        — good alternative
    """

    _BASE_URL = "https://api.groq.com/openai/v1/chat/completions"

    def __init__(self, api_key: str, model: str = "llama-3.3-70b-versatile"):
        self.api_key = api_key
        self.default_model = model

    def generate(self, prompt: str, model: str = None) -> str:
        """Single-turn generation with forced JSON output (used for column intelligence)."""
        model = model or self.default_model
        response = requests.post(
            self._BASE_URL,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "response_format": {"type": "json_object"},
                "temperature": 0.1,
                "max_tokens": 1024,
            },
            timeout=30,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    def chat_completion(self, messages: List[Dict], model: str = None) -> str:
        """Multi-turn chat with system/user/assistant messages (used for chatbot).
        Returns free-form text — no JSON forcing so the LLM can write naturally.
        """
        model = model or self.default_model
        response = requests.post(
            self._BASE_URL,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": messages,
                "temperature": 0.3,
                "max_tokens": 2048,
            },
            timeout=60,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
