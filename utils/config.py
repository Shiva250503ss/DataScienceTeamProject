# utils/config.py

import os
from dataclasses import dataclass
from dotenv import load_dotenv

# Load .env file before reading any env vars so API keys are available
load_dotenv()

@dataclass
class Config:
    # Database
    DATABASE_URL: str = os.getenv("DATABASE_URL", "postgresql://datapilot:datapilot123@localhost:5432/datapilot")
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379")
    QDRANT_URL: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    
    # LLM — primary: Gemini, fallback: Groq
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

config = Config()
