# agents/base.py

from abc import ABC, abstractmethod
from typing import Any, Dict
import pandas as pd
from utils.config import config


def _build_llm():
    """
    Build the pipeline LLM with Gemini 2.5 Flash as primary and Groq as fallback.

    Strategy:
      1. Try Gemini 2.5 Flash (google-generativeai via langchain-google-genai).
      2. If Gemini raises ANY exception (rate limit, quota, network, etc.),
         LangChain's with_fallbacks() automatically retries with Groq.
      3. If neither key is configured, raises a clear error on first LLM call.
    """
    llm = None

    if config.GEMINI_API_KEY:
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            gemini = ChatGoogleGenerativeAI(
                model=config.GEMINI_MODEL,
                google_api_key=config.GEMINI_API_KEY,
                temperature=0.3,
                max_output_tokens=2048,
                convert_system_message_to_human=True,  # Gemini has no system role
            )
            if config.GROQ_API_KEY:
                from langchain_groq import ChatGroq
                groq = ChatGroq(
                    api_key=config.GROQ_API_KEY,
                    model_name=config.GROQ_MODEL,
                )
                # Gemini first; on any error automatically switch to Groq
                llm = gemini.with_fallbacks([groq])
            else:
                llm = gemini
        except Exception:
            pass  # fall through to Groq-only

    if llm is None and config.GROQ_API_KEY:
        from langchain_groq import ChatGroq
        llm = ChatGroq(
            api_key=config.GROQ_API_KEY,
            model_name=config.GROQ_MODEL,
        )

    return llm


class BaseAgent(ABC):
    """
    Base class for all AI agents in the DataPilot pipeline.

    Every agent inherits from this class and must implement the `execute()` method.
    Provides shared functionality:
      - LLM access via Gemini 2.5 Flash (primary) / Groq (automatic fallback)
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
        Uses Gemini 2.5 Flash; automatically falls back to Groq on any error.

        Args:
            prompt: The prompt string to send to the LLM.

        Returns:
            LLM response as a string.
        """
        if self.llm is None:
            return "LLM not configured — set GEMINI_API_KEY or GROQ_API_KEY in .env."
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
