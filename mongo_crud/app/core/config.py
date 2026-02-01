"""
Configuration module for the application.
Loads environment variables and provides settings using Pydantic BaseSettings.
Note: Uses .env file from project root (../../.env)
"""

import os
from pathlib import Path
from typing import List
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    Reads from root .env file for unified configuration.
    """
    
    # MongoDB Configuration
    mongodb_uri: str = Field(..., env="MONGODB_URI")
    database_name: str = Field(..., env="DATABASE_NAME")
    
    # Server Configuration
    port: int = Field(default=8000, env="PORT")
    environment: str = Field(default="development", env="ENVIRONMENT")
    
    # CORS Configuration
    allowed_origins: str = Field(
        default="http://localhost:3000,http://localhost:8000",
        env="ALLOWED_ORIGINS"
    )
    
    # API Configuration
    api_v1_prefix: str = "/api/v1"
    project_name: str = "MongoDB CRUD API"
    version: str = "1.0.0"
    description: str = "Generalized MongoDB CRUD API using FastAPI"
    
    # Database Connection Pool Settings
    max_pool_size: int = Field(default=10, env="MAX_POOL_SIZE")
    min_pool_size: int = Field(default=1, env="MIN_POOL_SIZE")
    max_idle_time_ms: int = Field(default=45000, env="MAX_IDLE_TIME_MS")
    
    class Config:
        # Use .env file from project root (two levels up from this file)
        env_file = Path(__file__).parent.parent.parent.parent / ".env"
        case_sensitive = False
        # Allow extra fields from .env (since we share with LLM Council config)
        extra = "ignore"
    
    @property
    def cors_origins(self) -> List[str]:
        """
        Parse comma-separated origins into a list.
        """
        return [origin.strip() for origin in self.allowed_origins.split(",")]


# Global settings instance
settings = Settings()
