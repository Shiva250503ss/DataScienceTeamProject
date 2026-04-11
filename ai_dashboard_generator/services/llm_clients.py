from __future__ import annotations

from typing import List, Dict, Optional
import requests


class GeminiClient:
    """
    Google Gemini AI client (primary LLM) with automatic Groq fallback.
    Uses the new google-genai SDK (google.genai).

    Primary model : gemini-2.5-flash
    Fallback      : GroqClient — triggered on ANY Gemini error (rate limit,
                    quota exceeded, network timeout, invalid response, etc.)
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.5-flash",
        fallback_client: Optional["GroqClient"] = None,
    ):
        self.api_key = api_key
        self.default_model = model
        self._fallback = fallback_client

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _client(self):
        """Return a configured google.genai Client."""
        from google import genai  # new google-genai SDK
        return genai.Client(api_key=self.api_key)

    @staticmethod
    def _build_contents(messages: List[Dict]) -> tuple:
        """
        Convert OpenAI-style messages → (system_instruction, contents list).

        New SDK format:
          contents = [{"role": "user"|"model", "parts": [{"text": "..."}]}, ...]
        System messages → system_instruction string (passed separately).
        """
        system_parts: List[str] = []
        contents: List[Dict] = []

        for msg in messages:
            role    = msg.get("role", "")
            content = msg.get("content", "")
            if role == "system":
                system_parts.append(content)
            elif role == "user":
                contents.append({"role": "user",  "parts": [{"text": content}]})
            elif role in ("assistant", "model"):
                contents.append({"role": "model", "parts": [{"text": content}]})

        system_instruction = "\n\n".join(system_parts) if system_parts else None
        return system_instruction, contents

    # ── Public API ────────────────────────────────────────────────────────────

    def generate(self, prompt: str, model: str = None) -> str:
        """Single-turn generation (used for column intelligence — expects JSON)."""
        try:
            from google import genai
            from google.genai import types
            client = self._client()
            cfg = types.GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=1024,
            )
            # Try JSON mime type first; fall back to plain text if unsupported
            try:
                cfg_json = types.GenerateContentConfig(
                    temperature=0.1,
                    max_output_tokens=1024,
                    response_mime_type="application/json",
                )
                response = client.models.generate_content(
                    model=model or self.default_model,
                    contents=prompt,
                    config=cfg_json,
                )
            except Exception:
                response = client.models.generate_content(
                    model=model or self.default_model,
                    contents=prompt,
                    config=cfg,
                )
            return response.text
        except Exception as exc:
            if self._fallback is not None:
                return self._fallback.generate(prompt, model=None)
            raise RuntimeError(f"Gemini generate failed and no fallback: {exc}") from exc

    def chat_completion(self, messages: List[Dict], model: str = None) -> str:
        """Multi-turn chat with system/user/assistant messages (used for chatbot)."""
        try:
            from google import genai
            from google.genai import types
            client = self._client()
            system_instruction, contents = self._build_contents(messages)

            cfg = types.GenerateContentConfig(
                temperature=0.3,
                max_output_tokens=2048,
                system_instruction=system_instruction,
            )
            response = client.models.generate_content(
                model=model or self.default_model,
                contents=contents,
                config=cfg,
            )
            return response.text
        except Exception as exc:
            if self._fallback is not None:
                return self._fallback.chat_completion(messages, model=None)
            raise RuntimeError(f"Gemini chat failed and no fallback: {exc}") from exc


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
