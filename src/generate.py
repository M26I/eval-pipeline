"""Generation module for RAG responses."""

from src.config import settings
from src.providers import OllamaProvider

# Prompt template for in-context generation
GENERATION_PROMPT = """You are a helpful assistant answering questions based on provided documents.

CONTEXT:
{context}

INSTRUCTIONS:
- Answer the user's question ONLY using information from the context above.
- If the context is insufficient to answer the question, say: "I don't have enough information to answer that based on the documents."
- Be concise and direct.

QUESTION:
{question}

ANSWER:"""


def generate_answer(query: str, retrieved_chunks: list[dict]) -> dict:
    """Generate an answer from retrieved chunks with abstention gating.
    
    Abstains (returns "I don't know") if:
    - No chunks retrieved, OR
    - Best (lowest) distance exceeds abstention_threshold
    
    Otherwise, builds in-context prompt and calls LLM.
    
    Args:
        query: Original user query
        retrieved_chunks: List of dicts from retrieve() with keys: text, doc_id, distance
        
    Returns:
        Dict with keys:
        - answer: Generated answer string (or abstention message)
        - abstained: Boolean flag
        - sources: List of doc_ids used (empty if abstained)
    """
    # Check abstention conditions
    if not retrieved_chunks or retrieved_chunks[0]["distance"] > settings.abstention_threshold:
        return {
            "answer": "I don't know based on the available documents.",
            "abstained": True,
            "sources": [],
        }
    
    # Build context from chunks
    context = "\n\n".join(
        f"[{chunk['doc_id']}]\n{chunk['text']}"
        for chunk in retrieved_chunks
    )
    
    # Build prompt and generate
    prompt = GENERATION_PROMPT.format(context=context, question=query)
    provider = OllamaProvider(model=settings.llm_model)
    answer = provider.generate(prompt)
    
    # Extract unique doc_ids from sources
    sources = list(set(chunk["doc_id"] for chunk in retrieved_chunks))
    
    return {
        "answer": answer.strip(),
        "abstained": False,
        "sources": sources,
    }
