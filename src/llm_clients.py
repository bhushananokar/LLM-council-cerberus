"""
LLM API Clients
===============
Client wrappers for Groq and Google AI APIs.

Supported Models:
- Groq: llama-3.3-70b-versatile, qwen-2.5-72b-instruct
- Google AI: gemini-2.0-flash-exp

Features:
- Unified interface for different LLM providers
- Automatic retry logic
- Error handling and logging
- Token usage tracking
- JSON response formatting
"""

import json
import logging
import time
from typing import Dict, Any, Optional, Union
from abc import ABC, abstractmethod

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    Groq = None

try:
    import google.generativeai as genai
    GOOGLE_AI_AVAILABLE = True
except ImportError:
    GOOGLE_AI_AVAILABLE = False
    genai = None

from config.settings import settings

logger = logging.getLogger(__name__)


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients."""
    
    def __init__(self, api_key: str, provider: str):
        """
        Initialize base LLM client.
        
        Args:
            api_key: API key for the provider
            provider: Provider name (e.g., "groq", "google_ai")
        """
        self.api_key = api_key
        self.provider = provider
        self.max_retries = settings.MAX_RETRIES
        self.retry_delay = settings.RETRY_DELAY_SECONDS
        
        logger.info(f"Initialized {provider} client")
    
    @abstractmethod
    def generate(self,
                model: str,
                system_prompt: str,
                user_prompt: str,
                max_tokens: int,
                temperature: float,
                response_format: str) -> Union[str, Dict]:
        """
        Generate response from LLM.
        
        Args:
            model: Model identifier
            system_prompt: System prompt defining behavior
            user_prompt: User prompt with context
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            response_format: "json" or "text"
            
        Returns:
            Generated response (string or dict)
        """
        pass
    
    def _retry_with_exponential_backoff(self, func, *args, **kwargs):
        """
        Retry function with exponential backoff.
        
        Args:
            func: Function to retry
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Function result
            
        Raises:
            Exception: If all retries fail
        """
        for attempt in range(self.max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if attempt == self.max_retries - 1:
                    logger.error(f"[{self.provider}] All {self.max_retries} retries failed: {str(e)}")
                    raise
                
                delay = self.retry_delay * (2 ** attempt)
                logger.warning(f"[{self.provider}] Attempt {attempt + 1} failed: {str(e)}. Retrying in {delay}s...")
                time.sleep(delay)


class GroqClient(BaseLLMClient):
    """Client for Groq API (Llama, Qwen models)."""
    
    def __init__(self, api_key: str):
        """
        Initialize Groq client.
        
        Args:
            api_key: Groq API key
        """
        super().__init__(api_key, "groq")
        
        if not GROQ_AVAILABLE:
            raise ImportError("Groq package not installed. Install with: pip install groq")
        
        self.client = Groq(api_key=api_key)
        logger.info("Groq client initialized successfully")
    
    def generate(self,
                model: str,
                system_prompt: str,
                user_prompt: str,
                max_tokens: int = 400,
                temperature: float = 0.1,
                response_format: str = "json") -> Union[str, Dict]:
        """
        Generate response using Groq API.
        
        Args:
            model: Model name (e.g., "llama-3.3-70b-versatile")
            system_prompt: System prompt
            user_prompt: User prompt
            max_tokens: Max tokens to generate
            temperature: Sampling temperature
            response_format: "json" or "text"
            
        Returns:
            Generated response
        """
        logger.debug(f"[Groq] Generating with model: {model}")
        
        def _call_api():
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            # Build completion parameters
            completion_params = {
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature
            }
            
            # Add response format if JSON requested
            if response_format == "json":
                completion_params["response_format"] = {"type": "json_object"}
            
            response = self.client.chat.completions.create(**completion_params)
            
            return response
        
        try:
            # Call API with retry logic
            response = self._retry_with_exponential_backoff(_call_api)
            
            # Extract content
            content = response.choices[0].message.content
            
            # Log token usage
            if hasattr(response, 'usage'):
                logger.debug(f"[Groq] Tokens used: {response.usage.total_tokens} "
                           f"(input: {response.usage.prompt_tokens}, output: {response.usage.completion_tokens})")
            
            # Parse JSON if requested
            if response_format == "json":
                try:
                    parsed = json.loads(content)
                    # Add token usage to response
                    if hasattr(response, 'usage'):
                        parsed['_tokens_used'] = response.usage.total_tokens
                    return parsed
                except json.JSONDecodeError as e:
                    logger.error(f"[Groq] Failed to parse JSON response: {str(e)}")
                    # Return as dict with error
                    return {
                        "error": "json_parse_failed",
                        "raw_content": content,
                        "_tokens_used": response.usage.total_tokens if hasattr(response, 'usage') else 0
                    }
            
            return content
            
        except Exception as e:
            logger.error(f"[Groq] API call failed: {str(e)}")
            raise
    
    def chat_completion(self,
                       model: str,
                       system_prompt: str,
                       user_prompt: str,
                       max_tokens: int = 400,
                       temperature: float = 0.1,
                       response_format: str = "json") -> Union[str, Dict]:
        """
        Alias for generate method (for backward compatibility).
        
        Args:
            model: Model name
            system_prompt: System prompt
            user_prompt: User prompt
            max_tokens: Max tokens
            temperature: Temperature
            response_format: Response format
            
        Returns:
            Generated response
        """
        return self.generate(model, system_prompt, user_prompt, max_tokens, temperature, response_format)


class GoogleAIClient(BaseLLMClient):
    """Client for Google AI Studio API (Gemini models)."""
    
    def __init__(self, api_key: str):
        """
        Initialize Google AI client.
        
        Args:
            api_key: Google AI API key
        """
        super().__init__(api_key, "google_ai")
        
        if not GOOGLE_AI_AVAILABLE:
            raise ImportError("Google Generative AI package not installed. Install with: pip install google-generativeai")
        
        # Configure API
        genai.configure(api_key=api_key)
        
        # Model cache
        self._model_cache = {}
        
        logger.info("Google AI client initialized successfully")
    
    def _get_model(self, model_name: str):
        """
        Get or create model instance.
        
        Args:
            model_name: Model identifier
            
        Returns:
            GenerativeModel instance
        """
        if model_name not in self._model_cache:
            self._model_cache[model_name] = genai.GenerativeModel(model_name)
        return self._model_cache[model_name]
    
    def generate(self,
                model: str,
                system_prompt: str,
                user_prompt: str,
                max_tokens: int = 400,
                temperature: float = 0.1,
                response_format: str = "json") -> Union[str, Dict]:
        """
        Generate response using Google AI API.
        
        Args:
            model: Model name (e.g., "gemini-2.0-flash-exp")
            system_prompt: System prompt
            user_prompt: User prompt
            max_tokens: Max tokens to generate
            temperature: Sampling temperature
            response_format: "json" or "text"
            
        Returns:
            Generated response
        """
        logger.debug(f"[Google AI] Generating with model: {model}")
        
        def _call_api():
            # Get model instance
            model_instance = self._get_model(model)
            
            # Combine system prompt and user prompt
            # Google AI doesn't have separate system/user roles in the same way
            combined_prompt = f"{system_prompt}\n\n{user_prompt}"
            
            # Build generation config
            generation_config = genai.GenerationConfig(
                max_output_tokens=max_tokens,
                temperature=temperature
            )
            
            # Add JSON response MIME type if requested
            if response_format == "json":
                generation_config.response_mime_type = "application/json"
            
            # Generate content
            response = model_instance.generate_content(
                combined_prompt,
                generation_config=generation_config
            )
            
            return response
        
        try:
            # Call API with retry logic
            response = self._retry_with_exponential_backoff(_call_api)
            
            # Extract content
            content = response.text
            
            # Log token usage (if available)
            if hasattr(response, 'usage_metadata'):
                total_tokens = (response.usage_metadata.prompt_token_count + 
                              response.usage_metadata.candidates_token_count)
                logger.debug(f"[Google AI] Tokens used: {total_tokens} "
                           f"(input: {response.usage_metadata.prompt_token_count}, "
                           f"output: {response.usage_metadata.candidates_token_count})")
            
            # Parse JSON if requested
            if response_format == "json":
                try:
                    parsed = json.loads(content)
                    # Add token usage if available
                    if hasattr(response, 'usage_metadata'):
                        parsed['_tokens_used'] = (response.usage_metadata.prompt_token_count + 
                                                 response.usage_metadata.candidates_token_count)
                    return parsed
                except json.JSONDecodeError as e:
                    logger.error(f"[Google AI] Failed to parse JSON response: {str(e)}")
                    # Return as dict with error
                    return {
                        "error": "json_parse_failed",
                        "raw_content": content,
                        "_tokens_used": 0
                    }
            
            return content
            
        except Exception as e:
            logger.error(f"[Google AI] API call failed: {str(e)}")
            raise
    
    def generate_content(self,
                        model: str,
                        system_prompt: str,
                        user_prompt: str,
                        max_tokens: int = 400,
                        temperature: float = 0.1,
                        response_format: str = "json") -> Union[str, Dict]:
        """
        Alias for generate method (for backward compatibility).
        
        Args:
            model: Model name
            system_prompt: System prompt
            user_prompt: User prompt
            max_tokens: Max tokens
            temperature: Temperature
            response_format: Response format
            
        Returns:
            Generated response
        """
        return self.generate(model, system_prompt, user_prompt, max_tokens, temperature, response_format)


# Client factory
def create_llm_client(provider: str, api_key: str) -> BaseLLMClient:
    """
    Create LLM client based on provider.
    
    Args:
        provider: Provider name ("groq" or "google_ai")
        api_key: API key
        
    Returns:
        LLM client instance
        
    Raises:
        ValueError: If provider is unknown
    """
    if provider.lower() == "groq":
        return GroqClient(api_key)
    elif provider.lower() in ["google_ai", "google", "gemini"]:
        return GoogleAIClient(api_key)
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")


# Convenience functions for common operations
def call_groq_model(model: str,
                   system_prompt: str,
                   user_prompt: str,
                   api_key: str = None,
                   max_tokens: int = 400,
                   temperature: float = 0.1,
                   response_format: str = "json") -> Union[str, Dict]:
    """
    Quick call to Groq model.
    
    Args:
        model: Model name
        system_prompt: System prompt
        user_prompt: User prompt
        api_key: API key (defaults to settings)
        max_tokens: Max tokens
        temperature: Temperature
        response_format: Response format
        
    Returns:
        Generated response
    """
    api_key = api_key or settings.GROQ_API_KEY
    client = GroqClient(api_key)
    return client.generate(model, system_prompt, user_prompt, max_tokens, temperature, response_format)


def call_google_ai_model(model: str,
                        system_prompt: str,
                        user_prompt: str,
                        api_key: str = None,
                        max_tokens: int = 400,
                        temperature: float = 0.1,
                        response_format: str = "json") -> Union[str, Dict]:
    """
    Quick call to Google AI model.
    
    Args:
        model: Model name
        system_prompt: System prompt
        user_prompt: User prompt
        api_key: API key (defaults to settings)
        max_tokens: Max tokens
        temperature: Temperature
        response_format: Response format
        
    Returns:
        Generated response
    """
    api_key = api_key or settings.GOOGLE_AI_API_KEY
    client = GoogleAIClient(api_key)
    return client.generate(model, system_prompt, user_prompt, max_tokens, temperature, response_format)


# Test connection functions
def test_groq_connection(api_key: str = None) -> bool:
    """
    Test Groq API connection.
    
    Args:
        api_key: API key (defaults to settings)
        
    Returns:
        True if connection successful
    """
    try:
        api_key = api_key or settings.GROQ_API_KEY
        client = GroqClient(api_key)
        
        # Simple test call
        response = client.generate(
            model="llama-3.3-70b-versatile",
            system_prompt="You are a test assistant.",
            user_prompt="Say 'hello'",
            max_tokens=10,
            temperature=0.1,
            response_format="text"
        )
        
        logger.info("Groq connection test successful")
        return True
        
    except Exception as e:
        logger.error(f"Groq connection test failed: {str(e)}")
        return False


def test_google_ai_connection(api_key: str = None) -> bool:
    """
    Test Google AI API connection.
    
    Args:
        api_key: API key (defaults to settings)
        
    Returns:
        True if connection successful
    """
    try:
        api_key = api_key or settings.GOOGLE_AI_API_KEY
        client = GoogleAIClient(api_key)
        
        # Simple test call
        response = client.generate(
            model="gemini-2.0-flash-exp",
            system_prompt="You are a test assistant.",
            user_prompt="Say 'hello'",
            max_tokens=10,
            temperature=0.1,
            response_format="text"
        )
        
        logger.info("Google AI connection test successful")
        return True
        
    except Exception as e:
        logger.error(f"Google AI connection test failed: {str(e)}")
        return False


def test_all_connections() -> Dict[str, bool]:
    """
    Test all LLM API connections.
    
    Returns:
        Dictionary with test results
    """
    results = {
        "groq": test_groq_connection(),
        "google_ai": test_google_ai_connection()
    }
    
    logger.info(f"Connection test results: {results}")
    return results