# utils/config.py

import os
from dataclasses import dataclass
from dotenv import load_dotenv

# Load .env file before reading any env vars
load_dotenv()

@dataclass
class Config:
    # Database
    DATABASE_URL: str = os.getenv("DATABASE_URL", "postgresql://datapilot:datapilot123@localhost:5432/datapilot")
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379")
    QDRANT_URL: str = os.getenv("QDRANT_URL", "http://localhost:6333")

    # LLM — primary: Mistral-7B-Instruct served locally by Ollama.
    # No API key needed; runs fully offline once the model is pulled:
    #   ollama pull mistral:7b-instruct
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "mistral:7b-instruct")

    # Optional cloud fallbacks (legacy) — only used if keys are set in .env.
    # Leave blank for a fully local, key-free deployment.
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    # ML Settings
    CV_FOLDS: int = int(os.getenv("CV_FOLDS", "5"))
    MAX_UPLOAD_SIZE_MB: int = int(os.getenv("MAX_UPLOAD_SIZE_MB", "100"))

    # Paths
    PPO_MODEL_PATH: str = os.getenv("PPO_MODEL_PATH", "./rl_selector/models")
    UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", "./uploads")

    # RAG (Retrieval-Augmented Generation) settings
    # Embedding model runs locally via sentence-transformers (no API key).
    RAG_EMBEDDING_MODEL: str = os.getenv("RAG_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    RAG_RERANKER_MODEL: str = os.getenv("RAG_RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    RAG_COLLECTION: str = os.getenv("RAG_COLLECTION", "datapilot_knowledge")

config = Config()
