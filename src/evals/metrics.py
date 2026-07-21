"""Evaluation metrics (pure functions, no I/O or model calls)."""

from typing import Optional
import statistics

# Refusal phrases for soft abstention detection
# These patterns indicate the LLM declined to answer even though retrieval returned results
REFUSAL_PHRASES = [
    "i don't know",
    "i do not know",
    "don't have enough information",
    "do not have enough information",
    "not enough information",
    "does not mention",
    "doesn't mention",
    "no mention of",
    "is unknown",
    "cannot answer",
    "can't answer",
    "not specified in",
    "not provided in",
    "do not provide",
    "does not provide",
    "not provide information",
    "no information available",
    "there is no information",
]


def is_soft_abstention(answer: str) -> bool:
    """Detect if answer text contains a soft refusal pattern.
    
    This is a deliberately simple heuristic that checks for common phrases
    indicating the LLM declined to answer (e.g. "I don't know", "not enough
    information"). The phrase list is intentionally general to apply across
    diverse domains; residual false negatives (missed refusals) are expected,
    especially for domain-specific or creative phrasings.
    
    Known limitations: false positives/negatives are possible if these phrases
    appear in valid factual context (e.g., "The identity is unknown" when citing
    the document accurately). For production, consider more robust NLI-based or
    LLM-graded approaches.
    
    Args:
        answer: Generated answer string
        
    Returns:
        True if answer matches a refusal pattern (case-insensitive), False otherwise.
    """
    answer_lower = answer.lower()
    return any(phrase in answer_lower for phrase in REFUSAL_PHRASES)


def retrieval_hit_rate(retrieved_ids: list[str], relevant_ids: list[str]) -> bool:
    """Check if any relevant document ID was retrieved.
    
    Args:
        retrieved_ids: List of document IDs returned by retriever
        relevant_ids: List of ground-truth relevant document IDs
        
    Returns:
        True if intersection is non-empty, False otherwise.
        Edge case: if relevant_ids is empty, returns False (nothing to retrieve).
    """
    if not relevant_ids:
        return False
    return bool(set(retrieved_ids) & set(relevant_ids))


def abstention_correct(predicted_abstain: bool, should_abstain: bool) -> bool:
    """Check if abstention decision was correct.
    
    Args:
        predicted_abstain: Whether model abstained (returned "I don't know")
        should_abstain: Whether it should have abstained (ground truth)
        
    Returns:
        True if predicted matches ground truth, False otherwise.
    """
    return predicted_abstain == should_abstain


def faithfulness(answer: str, required_substrings: list[str]) -> float:
    """Measure faithfulness: fraction of required substrings present in answer.
    
    Comparison is case-insensitive. Non-abstention answers must contain
    required context to be considered faithful.
    
    Args:
        answer: Generated answer string
        required_substrings: List of substrings that should be present
        
    Returns:
        Float in [0.0, 1.0] = (substrings found) / (total required).
        Edge case: empty required_substrings returns 1.0 (deliberate choice:
        nothing required = trivially faithful; caller should validate answer
        is non-empty separately if needed).
    """
    if not required_substrings:
        return 1.0
    
    answer_lower = answer.lower()
    found = sum(1 for substring in required_substrings if substring.lower() in answer_lower)
    return found / len(required_substrings)


def aggregate_results(results: list[dict]) -> dict:
    """Aggregate per-question evaluation results into summary statistics.
    
    Args:
        results: List of dicts, each with keys:
            - retrieved_ids: list[str]
            - relevant_ids: list[str]
            - predicted_abstain: bool
            - should_abstain: bool
            - answer: str
            - required_substrings: list[str]
            - latency_ms: float
            
    Returns:
        Summary dict with keys:
            - hit_rate_pct: float (0-100)
            - abstention_precision: float or None (if no predicted abstentions)
            - abstention_recall: float or None (if no ground-truth abstentions)
            - mean_faithfulness: float (over non-abstained answers only)
            - mean_latency_ms: float
            - p95_latency_ms: float
            - total_queries: int
            - total_abstained: int
    """
    if not results:
        return {
            "hit_rate_pct": 0.0,
            "abstention_precision": None,
            "abstention_recall": None,
            "mean_faithfulness": 0.0,
            "mean_latency_ms": 0.0,
            "p95_latency_ms": 0.0,
            "total_queries": 0,
            "total_abstained": 0,
        }
    
    # Hit rate: computed only over questions with non-empty relevant_ids (retrieval questions).
    # Abstention questions (relevant_ids=[]) are excluded by design, since retrieval
    # was never meant to find anything for them. They are measured separately via
    # abstention precision/recall metrics.
    retrieval_questions = [r for r in results if r["relevant_ids"]]
    if retrieval_questions:
        hits = sum(
            retrieval_hit_rate(r["retrieved_ids"], r["relevant_ids"])
            for r in retrieval_questions
        )
        hit_rate_pct = (hits / len(retrieval_questions)) * 100
    else:
        hit_rate_pct = 0.0
    
    # Abstention metrics
    predicted_abstain = [r["predicted_abstain"] for r in results]
    should_abstain = [r["should_abstain"] for r in results]
    
    # Precision: of predicted abstentions, how many were correct?
    num_predicted_abstain = sum(predicted_abstain)
    if num_predicted_abstain > 0:
        correct_abstentions = sum(
            abstention_correct(r["predicted_abstain"], r["should_abstain"])
            and r["predicted_abstain"]
            for r in results
        )
        abstention_precision = correct_abstentions / num_predicted_abstain
    else:
        abstention_precision = None
    
    # Recall: of ground-truth abstentions, how many did we predict?
    num_should_abstain = sum(should_abstain)
    if num_should_abstain > 0:
        correct_abstentions = sum(
            abstention_correct(r["predicted_abstain"], r["should_abstain"])
            and r["should_abstain"]
            for r in results
        )
        abstention_recall = correct_abstentions / num_should_abstain
    else:
        abstention_recall = None
    
    # Faithfulness (only on non-abstained answers)
    non_abstained = [
        r for r in results if not r["predicted_abstain"]
    ]
    if non_abstained:
        faithfulness_scores = [
            faithfulness(r["answer"], r["required_substrings"])
            for r in non_abstained
        ]
        mean_faithfulness = sum(faithfulness_scores) / len(faithfulness_scores)
    else:
        mean_faithfulness = 0.0
    
    # Latency stats
    latencies = [r["latency_ms"] for r in results]
    mean_latency_ms = sum(latencies) / len(latencies) if latencies else 0.0
    # Below 20 samples, p95 is approximated by max as a deliberate simplification.
    p95_latency_ms = (
        statistics.quantiles(latencies, n=20)[18]  # 19th of 20 quantiles = p95
        if len(latencies) >= 20
        else max(latencies) if latencies else 0.0
    )
    
    return {
        "hit_rate_pct": round(hit_rate_pct, 2),
        "abstention_precision": round(abstention_precision, 3) if abstention_precision is not None else None,
        "abstention_recall": round(abstention_recall, 3) if abstention_recall is not None else None,
        "mean_faithfulness": round(mean_faithfulness, 3),
        "mean_latency_ms": round(mean_latency_ms, 2),
        "p95_latency_ms": round(p95_latency_ms, 2),
        "total_queries": len(results),
        "total_abstained": sum(predicted_abstain),
    }
