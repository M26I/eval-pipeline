"""Retrieval module for RAG.

Note: Uses chromadb 0.4.x API. If query() behavior changes in newer versions,
verify against https://docs.trychroma.com/api-reference/py-client#query
"""

from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer
from src.config import settings

# Module-level singleton: embedding model loaded once and reused across queries
_embedding_model = None


def _get_embedding_model() -> SentenceTransformer:
    """Lazy-load and cache embedding model (singleton pattern).
    
    Returns:
        SentenceTransformer instance, reused across all calls.
    """
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(settings.embedding_model)
    return _embedding_model


def retrieve(query: str, k: int = None) -> list[dict]:
    """Retrieve top-k chunks for query using cosine distance.
    
    Args:
        query: Query string to embed and search
        k: Number of results (defaults to settings.top_k)
    
    Returns:
        List of dicts with keys: text, doc_id, distance (cosine distance)
        Distance is explicit so it can be used for abstention gating.
    """
    if k is None:
        k = settings.top_k
    
    # Initialize client and use cached embedding model
    client = chromadb.PersistentClient(path=str(settings.chroma_path))
    embedder = _get_embedding_model()
    collection = client.get_collection(name="documents")
    
    # Embed query and retrieve
    query_embedding = embedder.encode(query).tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k
    )
    
    # Flatten results (chromadb wraps in lists)
    output = []
    for i in range(len(results["ids"][0])):
        output.append({
            "text": results["documents"][0][i],
            "doc_id": results["metadatas"][0][i]["doc_id"],
            "distance": results["distances"][0][i],
        })
    
    return output
