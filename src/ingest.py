"""Document ingestion and vectorization."""

from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer
from src.config import settings


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Split text into overlapping chunks.
    
    Note: This is a deliberate simplification using character-based splitting.
    For production, consider token-aware splitting or a library like langchain.
    
    Args:
        text: Input text to chunk
        chunk_size: Target chunk size in characters (~500)
        overlap: Overlap between consecutive chunks (~50)
        
    Returns:
        List of text chunks
    """
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i : i + chunk_size])
    return chunks


def ingest() -> int:
    """Ingest corpus from data/corpus/ into persistent Chroma collection.
    
    Reads all .txt and .md files, chunks them, embeds with sentence-transformers,
    and stores in Chroma with metadata (doc_id, chunk_index).
    
    Idempotent: resets collection on each run to avoid duplicates.
    
    Returns:
        Total number of chunks stored
    """
    # Initialize client and embedding model
    client = chromadb.PersistentClient(path=str(settings.chroma_path))
    embedder = SentenceTransformer(settings.embedding_model)
    
    # Reset collection for idempotency
    try:
        client.delete_collection(name="documents")
    except Exception:
        pass
    
    collection = client.get_or_create_collection(
        name="documents",
        metadata={"hnsw:space": "cosine"}
    )
    
    # Process all .txt and .md files
    corpus_path = Path(settings.corpus_path)
    corpus_path.mkdir(parents=True, exist_ok=True)
    
    total_chunks = 0
    for file_path in sorted(list(corpus_path.glob("*.txt")) + list(corpus_path.glob("*.md"))):
        doc_id = file_path.stem
        
        # Try UTF-8 first, then fall back to UTF-16 (Windows default)
        try:
            text = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = file_path.read_text(encoding="utf-16")
        
        chunks = chunk_text(text)
        
        for chunk_idx, chunk in enumerate(chunks):
            embedding = embedder.encode(chunk).tolist()
            collection.add(
                ids=[f"{doc_id}_{chunk_idx}"],
                embeddings=[embedding],
                metadatas=[{"doc_id": doc_id, "chunk_index": chunk_idx}],
                documents=[chunk]
            )
            total_chunks += 1
    
    return total_chunks


if __name__ == "__main__":
    count = ingest()
    print(f"✓ Stored {count} chunks in Chroma at {settings.chroma_path}")
