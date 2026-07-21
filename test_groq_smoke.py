#!/usr/bin/env python
"""Smoke test for GroqProvider.

To run:
  1. Set your Groq API key in .env or export GROQ_API_KEY=<your-key>
  2. Set LLM_PROVIDER=groq in .env or export LLM_PROVIDER=groq
  3. Run: python test_groq_smoke.py
"""

import os
from dotenv import load_dotenv

# Load .env first
load_dotenv()

from src.providers import get_provider

if __name__ == "__main__":
    print("Testing GroqProvider via get_provider()...\n")
    
    # Check if GROQ_API_KEY is set
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or api_key.strip() == "":
        print("ERROR: GROQ_API_KEY is not set in .env or environment.")
        print("Set it in .env or run: export GROQ_API_KEY=your-key-here")
        exit(1)
    
    # Set provider to groq
    os.environ["LLM_PROVIDER"] = "groq"
    print(f"LLM_PROVIDER: {os.getenv('LLM_PROVIDER')}")
    print(f"GROQ_API_KEY is set: {'*' * 8}\n")
    
    # Get provider and generate response
    try:
        provider = get_provider()
        print(f"Provider type: {type(provider).__name__}")
        print(f"Model: {provider.model}\n")
        
        print("Sending test prompt to Groq API...")
        response = provider.generate("Say hello in one word")
        print(f"Response: {response}\n")
        print("✓ GroqProvider smoke test passed!")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        exit(1)
