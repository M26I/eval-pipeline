"""LLM and embedding provider interfaces.

This module defines a provider abstraction that mirrors production multi-provider
architectures. The abstract LLMProvider interface enables swapping Ollama for
hosted models (OpenAI, Anthropic, etc.) without touching pipeline code.
"""

import os
from abc import ABC, abstractmethod
from typing import Literal
import ollama
from groq import Groq

# Load environment variables from .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


class LLMProvider(ABC):
    """Abstract LLM provider interface."""
    
    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate text from prompt.
        
        Args:
            prompt: Input prompt string
            
        Returns:
            Generated text response
        """
        pass


class OllamaProvider(LLMProvider):
    """Ollama-based LLM provider for offline inference."""
    
    def __init__(self, model: str = "llama3.2:3b"):
        """Initialize Ollama provider.
        
        Args:
            model: Ollama model identifier (default: llama3.2:3b)
        """
        self.model = model
    
    def generate(self, prompt: str) -> str:
        """Generate text using Ollama.
        
        Args:
            prompt: Input prompt string
            
        Returns:
            Generated text response from model
        """
        response = ollama.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            stream=False,
        )
        return response["message"]["content"]


class GroqProvider(LLMProvider):
    """Groq-hosted LLM provider using official groq SDK.
    
    Uses Groq's fast inference API. Requires GROQ_API_KEY environment variable.
    """
    
    def __init__(self, model: str = "llama-3.3-70b-versatile"):
        """Initialize Groq provider.
        
        Args:
            model: Groq-hosted model identifier
            
        Raises:
            ValueError: If GROQ_API_KEY environment variable is not set
        """
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY environment variable must be set to use GroqProvider. "
                "Set it in .env or export it in your shell."
            )
        self.client = Groq(api_key=api_key)
        self.model = model
    
    def generate(self, prompt: str) -> str:
        """Generate text using Groq API.
        
        Args:
            prompt: Input prompt string
            
        Returns:
            Generated text response from model
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )
        return response.choices[0].message.content


def get_provider(provider: Literal["ollama", "groq"] | None = None, 
                 ollama_model: str | None = None,
                 groq_model: str | None = None) -> LLMProvider:
    """Factory function to get the configured LLM provider.
    
    Selects provider based on LLM_PROVIDER env var ("ollama" or "groq"),
    defaulting to Ollama if unset. Reads model names from config or params.
    
    Args:
        provider: Override LLM_PROVIDER env var ("ollama" or "groq")
        ollama_model: Override Ollama model name
        groq_model: Override Groq model name
        
    Returns:
        Configured LLMProvider instance (OllamaProvider or GroqProvider)
        
    Raises:
        ValueError: If provider is not "ollama" or "groq"
        ValueError: If "groq" is selected but GROQ_API_KEY is not set
    """
    from src.config import settings
    
    # Determine which provider to use
    selected_provider = provider or os.getenv("LLM_PROVIDER", "ollama").lower()
    
    if selected_provider == "ollama":
        model = ollama_model or settings.llm_model
        return OllamaProvider(model=model)
    elif selected_provider == "groq":
        model = groq_model or settings.groq_model
        return GroqProvider(model=model)
    else:
        raise ValueError(
            f"Unknown LLM provider: {selected_provider}. "
            "Must be 'ollama' or 'groq'."
        )
