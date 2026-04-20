"""
Configuration management for chatbot service
"""
from pydantic import BaseModel
from typing import Optional
import os


class Settings:
    """Application settings loaded from environment variables"""
    
    def __init__(self):
        # Ollama Configuration
        self.ollama_model = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
        self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "paraphrase-multilingual-mpnet-base-v2")
        
        # Paths
        self.knowledge_base_path = os.getenv("KNOWLEDGE_BASE_PATH", "data/articles.json")
        
        # RAG Configuration  
        self.top_k_articles = int(os.getenv("TOP_K_ARTICLES", "3"))
        self.temperature = float(os.getenv("TEMPERATURE", "0.1"))  # LOWERED for better Arabic
        
        # Server Configuration
        self.host = os.getenv("HOST", "0.0.0.0")
        self.port = int(os.getenv("PORT", "8001"))


# Global settings instance
settings = Settings()
