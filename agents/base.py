# agents/base.py

from abc import ABC, abstractmethod
from typing import Any, Dict
import pandas as pd
from utils.config import config


def _build_llm():
    """
    Build the pipeline LLM — local-first, no API keys required.

    Strategy:
      1. PRIMARY: Mistral-7B-Instruct served locally by Ollama.
         Fully offline, no API key, no token limits. Requires:
             ollama pull mistral:7b-instruct
      2. OPTIONAL FALLBACKS: if GEMINI_API_KEY / GROQ_API_KEY are set in .env,
         they are attached via LangChain's with_fallbacks() so any Ollama
         error (server down, model not pulled) automatically retries in the
         cloud. Leave the keys blank for a fully local deployment.
    """
    # ── Optional cloud fallbacks (only if keys are configured) ────────────
    fallbacks = []
    if config.GEMINI_API_KEY:
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            fallbacks.append(ChatGoogleGenerativeAI(
                model=config.GEMINI_MODEL,
                google_api_key=config.GEMINI_API_KEY,
                temperature=0.3,
                max_output_tokens=2048,
                convert_system_message_to_human=True,  # Gemini has no system role
            ))
        except Exception:
            pass
    if config.GROQ_API_KEY:
        try:
            from langchain_groq import ChatGroq
            fallbacks.append(ChatGroq(
                api_key=config.GROQ_API_KEY,
                model_name=config.GROQ_MODEL,
            ))
        except Exception:
            pass

    # ── Primary: local Mistral via Ollama ──────────────────────────────────
    llm = None
    try:
        from langchain_community.chat_models import ChatOllama
        llm = ChatOllama(
            base_url=config.OLLAMA_BASE_URL,
            model=config.OLLAMA_MODEL,
            temperature=0.3,
            # Mistral-Instruct works best with a modest context; num_ctx can be
            # raised (e.g. 8192) if the host has enough RAM/VRAM.
            num_ctx=4096,
        )
    except Exception:
        pass  # langchain-community missing — fall through to cloud-only

    if llm is not None and fallbacks:
        return llm.with_fallbacks(fallbacks)
    if llm is not None:
        return llm
    # No Ollama available — use first configured cloud model (legacy mode)
    if fallbacks:
        primary, *rest = fallbacks
        return primary.with_fallbacks(rest) if rest else primary
    return None


class BaseAgent(ABC):
    """
    Base class for all AI agents in the DataPilot pipeline.

    Every agent inherits from this class and must implement the `execute()` method.
    Provides shared functionality:
      - LLM access via local Mistral-7B-Instruct on Ollama (primary),
        with optional Gemini/Groq cloud fallbacks if API keys are configured
      - Logging with agent name prefix
      - Standard execute interface that takes/returns pipeline state
    """

    def __init__(self, name: str):
        self.name = name
        self.llm = _build_llm()

    @abstractmethod
    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the agent's task and return updated state.

        Args:
            state: Dictionary containing the full pipeline state
                   (raw data, profile, cleaned data, features, models, etc.)

        Returns:
            Updated state dictionary with this agent's outputs added.
        """
        pass

    def ask_llm(self, prompt: str) -> str:
        """
        Query the LLM for reasoning/explanations.
        Uses local Mistral-7B-Instruct via Ollama; automatically falls back
        to Gemini/Groq only if those API keys are configured.

        Args:
            prompt: The prompt string to send to the LLM.

        Returns:
            LLM response as a string.
        """
        if self.llm is None:
            return ("LLM not configured — start Ollama (`ollama serve`) and pull the model "
                    "(`ollama pull mistral:7b-instruct`), or set a cloud API key in .env.")
        response = self.llm.invoke(prompt)
        return response.content
    
    def log(self, message: str):
        """
        Log agent activity with agent name prefix.

        Args:
            message: The log message to print.
        """
        try:
            print(f"[{self.name}] {message}")
        except UnicodeEncodeError:
            # Windows cp1252 can't handle some Unicode chars — strip them
            safe = message.encode('ascii', errors='replace').decode('ascii')
            print(f"[{self.name}] {safe}")
