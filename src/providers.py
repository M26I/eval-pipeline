"""LLM and embedding provider interfaces.

This module defines a provider abstraction that mirrors production multi-provider
architectures. The abstract LLMProvider interface enables swapping Ollama for
hosted models (OpenAI, Anthropic, etc.) without touching pipeline code.
"""

from abc import ABC, abstractmethod
import ollama


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
