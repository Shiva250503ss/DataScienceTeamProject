# agents/base.py

from abc import ABC, abstractmethod
from typing import Any, Dict
import pandas as pd
from langchain_groq import ChatGroq
from utils.config import config


class BaseAgent(ABC):
    """
    Base class for all AI agents in the DataPilot pipeline.

    Every agent inherits from this class and must implement the `execute()` method.
    Provides shared functionality:
      - LLM access via Groq (for AI-powered reasoning)
      - Logging with agent name prefix
      - Standard execute interface that takes/returns pipeline state
    """

    def __init__(self, name: str):
        self.name = name
        self.llm = ChatGroq(
            api_key=config.GROQ_API_KEY,
            model_name=config.GROQ_MODEL,
        )
    
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

        Args:
            prompt: The prompt string to send to the LLM.

        Returns:
            LLM response as a string.
        """
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
