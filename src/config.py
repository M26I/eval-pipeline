"""Configuration management."""

from pathlib import Path
from pydantic import BaseModel, Field


class Settings(BaseModel):
    """Evaluation pipeline configuration.
    
    All paths are resolved relative to the project root at import time.
    """
    
    llm_model: str = Field(
        default="llama3.2:3b",
        description="Ollama model identifier for generation"
    )
    groq_model: str = Field(
        default="llama-3.3-70b-versatile",
        description="Groq-hosted model identifier (used when llm_provider='groq')"
    )
    llm_provider: str = Field(
        default="ollama",
        description="LLM provider: 'ollama' (local) or 'groq' (hosted)"
    )
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="HuggingFace model identifier for embeddings"
    )
    top_k: int = Field(
        default=4,
        description="Number of top-k retrieval results"
    )
    abstention_threshold: float = Field(
        default=0.7,
        description="Cosine distance threshold for retrieval confidence (tunable)"
    )
    chroma_path: Path = Field(
        default_factory=lambda: Path("chroma_data"),
        description="Path to chromadb persistent storage"
    )
    corpus_path: Path = Field(
        default_factory=lambda: Path("data/corpus"),
        description="Path to input document corpus"
    )
    eval_set_path: Path = Field(
        default_factory=lambda: Path("data/eval_set.jsonl"),
        description="Path to evaluation dataset"
    )
    results_path: Path = Field(
        default_factory=lambda: Path("results"),
        description="Path to results directory (project root)"
    )
    
    class Config:
        """Pydantic config."""
        frozen = False


# Global settings instance
settings = Settings()
