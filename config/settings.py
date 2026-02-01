"""
Configuration Settings
======================
Central configuration for the LLM Council system.

All settings can be overridden via environment variables.
"""

import os
from pathlib import Path
from typing import Optional, List, Union
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import field_validator, Field
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        env_parse_none_str="None",
        extra="ignore"  # Ignore MongoDB and other extra variables
    )
    
    # ============================================================================
    # APPLICATION INFO
    # ============================================================================
    APP_NAME: str = "LLM Council - Supply Chain Security"
    APP_VERSION: str = "1.0.0"
    ENVIRONMENT: str = "development"
    
    # ============================================================================
    # API KEYS
    # ============================================================================
    GROQ_API_KEY: str = ""
    GOOGLE_AI_API_KEY: str = ""
    
    # ============================================================================
    # LLM MODELS
    # ============================================================================
    # Agent 1: Code Intelligence
    AGENT1_MODEL: str = "llama-3.3-70b-versatile"
    AGENT1_WEIGHT: float = 0.30
    
    # Agent 2: Threat Intelligence
    AGENT2_MODEL: str = "qwen-2.5-72b-instruct"
    AGENT2_WEIGHT: float = 0.35
    
    # Agent 3: Behavioral Intelligence
    AGENT3_MODEL: str = "gemini-2.0-flash-exp"
    AGENT3_WEIGHT: float = 0.35
    
    # Overseer Agent
    OVERSEER_MODEL: str = "gemini-2.0-flash-thinking-exp"
    OVERSEER_ENABLED: bool = True
    
    # ============================================================================
    # DEBATE SYSTEM PARAMETERS
    # ============================================================================
    # Debate configuration
    MAX_DEBATE_ROUNDS: int = 5
    CONSENSUS_THRESHOLD: float = 0.67  # 67% agreement needed
    ENABLE_REFLECTIONS: bool = True
    ENABLE_OVERSEER_INTERVENTION: bool = True
    
    # Intervention thresholds
    REASONING_QUALITY_THRESHOLD: float = 50.0  # Below this triggers intervention
    
    # ============================================================================
    # LLM GENERATION PARAMETERS
    # ============================================================================
    MAX_OUTPUT_TOKENS: int = 1000
    TEMPERATURE: float = 0.1
    
    # Enhanced tokens for debate phases
    MAX_TOKENS_PRESENTATION: int = 1500
    MAX_TOKENS_REFLECTION: int = 1000
    MAX_TOKENS_VOTE: int = 800
    
    # ============================================================================
    # CONSENSUS PARAMETERS
    # ============================================================================
    # Variance thresholds for agreement detection
    VARIANCE_THRESHOLD_STRONG: float = 10.0
    VARIANCE_THRESHOLD_MODERATE: float = 20.0
    
    # ============================================================================
    # RETRY & TIMEOUT
    # ============================================================================
    MAX_RETRIES: int = 3
    RETRY_DELAY_SECONDS: float = 2.0
    REQUEST_TIMEOUT_SECONDS: int = 60
    
    # ============================================================================
    # REDIS CACHE
    # ============================================================================
    CACHE_ENABLED: bool = True
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_PASSWORD: Optional[str] = None
    REDIS_DB: int = 0
    CACHE_TTL_SECONDS: int = 86400  # 24 hours
    
    # ============================================================================
    # PROMPTS
    # ============================================================================
    PROMPTS_DIR: str = "config/prompts"
    
    # ============================================================================
    # LOGGING
    # ============================================================================
    LOG_LEVEL: str = "INFO"
    LOG_FILE: Optional[str] = None
    
    # ============================================================================
    # API SERVER (FastAPI)
    # ============================================================================
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_RELOAD: bool = True
    API_WORKERS: int = 1
    
    # CORS
    CORS_ORIGINS: Union[str, List[str]] = "*"
    CORS_ALLOW_CREDENTIALS: bool = True
    CORS_ALLOW_METHODS: List[str] = ["*"]
    CORS_ALLOW_HEADERS: List[str] = ["*"]
    
    @field_validator("CORS_ORIGINS", mode="after")
    @classmethod
    def parse_cors_origins(cls, v):
        """Parse CORS origins from string or list."""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v
    
    # ============================================================================
    # RATE LIMITING
    # ============================================================================
    RATE_LIMIT_ENABLED: bool = False
    RATE_LIMIT_REQUESTS: int = 100
    RATE_LIMIT_PERIOD_SECONDS: int = 60
    
    # ============================================================================
    # DATABASE (Optional - for future use)
    # ============================================================================
    DATABASE_ENABLED: bool = False
    DATABASE_URL: Optional[str] = None
    
    # ============================================================================
    # BLOCKCHAIN (Optional - for future use)
    # ============================================================================
    BLOCKCHAIN_ENABLED: bool = False
    POLYGON_RPC_URL: Optional[str] = None
    PRIVATE_KEY: Optional[str] = None
    
    # ============================================================================
    # FEATURE FLAGS
    # ============================================================================
    ENABLE_DEBATE: bool = True
    ENABLE_CACHE: bool = True
    ENABLE_PARALLEL_EXECUTION: bool = True
    
    # ============================================================================
    # PATHS
    # ============================================================================
    @property
    def BASE_DIR(self) -> Path:
        return Path(__file__).parent.parent
    
    @property
    def CONFIG_DIR(self) -> Path:
        return self.BASE_DIR / "config"
    
    @property
    def PROMPTS_PATH(self) -> Path:
        return Path(self.PROMPTS_DIR)
    
    # ============================================================================
    # VALIDATION
    # ============================================================================
    
    def validate_settings(self) -> bool:
        """
        Validate critical settings.
        
        Returns:
            True if valid
            
        Raises:
            ValueError: If critical settings are missing
        """
        errors = []
        
        # Check API keys
        if not self.GROQ_API_KEY:
            errors.append("GROQ_API_KEY is required")
        
        if not self.GOOGLE_AI_API_KEY:
            errors.append("GOOGLE_AI_API_KEY is required")
        
        # Check agent weights sum to 1.0
        total_weight = self.AGENT1_WEIGHT + self.AGENT2_WEIGHT + self.AGENT3_WEIGHT
        if not 0.99 <= total_weight <= 1.01:  # Allow small floating point errors
            errors.append(f"Agent weights must sum to 1.0 (current: {total_weight})")
        
        # Check variance thresholds
        if self.VARIANCE_THRESHOLD_STRONG >= self.VARIANCE_THRESHOLD_MODERATE:
            errors.append("VARIANCE_THRESHOLD_STRONG must be less than VARIANCE_THRESHOLD_MODERATE")
        
        # Check prompts directory exists
        if not self.PROMPTS_PATH.exists():
            errors.append(f"Prompts directory not found: {self.PROMPTS_PATH}")
        
        if errors:
            raise ValueError("Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors))
        
        return True
    
    def get_model_config(self, agent_name: str) -> dict:
        """
        Get model configuration for specific agent.
        
        Args:
            agent_name: Agent identifier
            
        Returns:
            Dictionary with model config
        """
        configs = {
            "code_intelligence": {
                "model": self.AGENT1_MODEL,
                "weight": self.AGENT1_WEIGHT,
                "provider": "groq"
            },
            "threat_intelligence": {
                "model": self.AGENT2_MODEL,
                "weight": self.AGENT2_WEIGHT,
                "provider": "groq"
            },
            "behavioral_intelligence": {
                "model": self.AGENT3_MODEL,
                "weight": self.AGENT3_WEIGHT,
                "provider": "google_ai"
            }
        }
        
        return configs.get(agent_name, {})
    
    def to_dict(self) -> dict:
        """
        Convert settings to dictionary (excluding sensitive data).
        
        Returns:
            Dictionary with non-sensitive settings
        """
        return {
            "app_name": self.APP_NAME,
            "app_version": self.APP_VERSION,
            "environment": self.ENVIRONMENT,
            "models": {
                "agent1": self.AGENT1_MODEL,
                "agent2": self.AGENT2_MODEL,
                "agent3": self.AGENT3_MODEL
            },
            "weights": {
                "agent1": self.AGENT1_WEIGHT,
                "agent2": self.AGENT2_WEIGHT,
                "agent3": self.AGENT3_WEIGHT
            },
            "generation": {
                "max_tokens": self.MAX_OUTPUT_TOKENS,
                "temperature": self.TEMPERATURE
            },
            "consensus": {
                "variance_strong": self.VARIANCE_THRESHOLD_STRONG,
                "variance_moderate": self.VARIANCE_THRESHOLD_MODERATE
            },
            "cache": {
                "enabled": self.CACHE_ENABLED,
                "ttl_seconds": self.CACHE_TTL_SECONDS
            },
            "features": {
                "debate": self.ENABLE_DEBATE,
                "cache": self.ENABLE_CACHE,
                "parallel_execution": self.ENABLE_PARALLEL_EXECUTION
            }
        }
    
    def __repr__(self) -> str:
        """String representation."""
        return f"<Settings env={self.ENVIRONMENT} version={self.APP_VERSION}>"


# ============================================================================
# SINGLETON INSTANCE
# ============================================================================

# Create global settings instance
settings = Settings()

# Validate on import (can be disabled for testing)
if os.getenv("SKIP_SETTINGS_VALIDATION", "false").lower() != "true":
    try:
        settings.validate_settings()
    except ValueError as e:
        import sys
        print(f"⚠️  Configuration Error: {str(e)}", file=sys.stderr)
        if settings.ENVIRONMENT == "production":
            sys.exit(1)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_settings() -> Settings:
    """
    Get settings instance.
    
    Returns:
        Settings singleton
    """
    return settings


def reload_settings():
    """Reload settings from environment."""
    global settings
    load_dotenv(override=True)
    settings = Settings()
    settings.validate_settings()


def print_settings():
    """Print current settings (non-sensitive)."""
    import json
    print(json.dumps(settings.to_dict(), indent=2))


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "settings",
    "Settings",
    "get_settings",
    "reload_settings",
    "print_settings"
]