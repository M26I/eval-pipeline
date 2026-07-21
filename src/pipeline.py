"""Main RAG pipeline orchestration."""

import time
import json
import sys
from src.retrieve import retrieve
from src.generate import generate_answer


def answer(query: str) -> dict:
    """Answer a query using RAG pipeline (retrieve → generate).
    
    Single entry point for the evaluation harness.
    
    Args:
        query: User query string
        
    Returns:
        Dict with keys:
        - answer: Generated answer
        - abstained: Boolean flag
        - sources: List of doc_ids
        - retrieved_chunks: Raw retrieval results
        - latency_ms: Total pipeline latency in milliseconds
    """
    start_time = time.time()
    
    # Retrieve
    retrieved_chunks = retrieve(query)
    
    # Generate
    result = generate_answer(query, retrieved_chunks)
    
    # Add metadata
    result["retrieved_chunks"] = retrieved_chunks
    result["latency_ms"] = round((time.time() - start_time) * 1000, 2)
    
    return result


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.pipeline <query>")
        print("Example: python -m src.pipeline 'What does the document say?'")
        sys.exit(1)
    
    query = " ".join(sys.argv[1:])
    result = answer(query)
    
    # Pretty print result
    print("\n" + "="*60)
    print(f"QUERY: {query}")
    print("="*60)
    print(f"\nABSTAINED: {result['abstained']}")
    print(f"\nANSWER:\n{result['answer']}")
    print(f"\nSOURCES: {', '.join(result['sources']) if result['sources'] else 'None'}")
    print(f"LATENCY: {result['latency_ms']}ms")
    print(f"RETRIEVED: {len(result['retrieved_chunks'])} chunks")
    print("\nDETAILS:")
    for i, chunk in enumerate(result["retrieved_chunks"], 1):
        print(f"  [{i}] {chunk['doc_id']} (distance={chunk['distance']:.3f})")
    print("="*60 + "\n")
