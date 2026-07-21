"""Evaluation runner.

Loads evaluation dataset (JSONL), runs pipeline on each question, and outputs
summary metrics + per-question results.

Evaluation set format (data/eval_set.jsonl):
  {"query": "...", "relevant_ids": [...], "required_substrings": [...]}

Each question can be:
- Answerable: relevant_ids is non-empty, system should retrieve and generate
- Abstention: relevant_ids is empty, system should refuse (abstain)
"""

import json
import sys
from pathlib import Path
from datetime import datetime, timezone
from src.config import settings
from src.pipeline import answer
from src.evals.metrics import aggregate_results, is_soft_abstention


def run_evaluation() -> dict:
    """Run evaluation on all questions in eval_set.jsonl.
    
    Returns:
        Dict with keys:
        - summary: aggregated metrics from aggregate_results()
        - per_question_results: list of per-question evaluation dicts
        - eval_set_path: path to input eval set
        - timestamp: ISO 8601 timestamp of run
    """
    eval_path = Path(settings.eval_set_path)
    
    if not eval_path.exists():
        raise FileNotFoundError(f"Evaluation set not found: {eval_path}")
    
    # Load evaluation set
    questions = []
    with open(eval_path) as f:
        for line_num, line in enumerate(f, 1):
            try:
                questions.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at line {line_num}: {e}")
    
    if not questions:
        raise ValueError("Evaluation set is empty")
    
    print(f"\nLoading {len(questions)} questions from {eval_path}...\n")
    
    # Run evaluation
    per_question_results = []
    for i, q in enumerate(questions, 1):
        query = q.get("query", "")
        relevant_ids = q.get("relevant_ids", [])
        required_substrings = q.get("required_substrings", [])
        should_abstain = len(relevant_ids) == 0
        
        # Call pipeline
        pipeline_result = answer(query)
        
        # Compute effective abstention: hard gate OR soft LLM refusal
        hard_abstained = pipeline_result["abstained"]
        soft_abstained = is_soft_abstention(pipeline_result["answer"])
        effective_abstain = hard_abstained or soft_abstained
        
        # Extract retrieval info
        retrieved_ids = [chunk["doc_id"] for chunk in pipeline_result["retrieved_chunks"]]
        
        # Build evaluation record
        eval_record = {
            "query": query,
            "answer": pipeline_result["answer"],
            "predicted_abstain": effective_abstain,
            "hard_abstained": hard_abstained,
            "soft_abstained": soft_abstained,
            "should_abstain": should_abstain,
            "retrieved_ids": retrieved_ids,
            "relevant_ids": relevant_ids,
            "required_substrings": required_substrings,
            "sources": pipeline_result["sources"],
            "latency_ms": pipeline_result["latency_ms"],
        }
        per_question_results.append(eval_record)
        
        # Progress indicator
        print(f"[{i}/{len(questions)}] {query[:60]:<60} | "
              f"retrieved={len(retrieved_ids)} | "
              f"abstain={pipeline_result['abstained']} | "
              f"latency={pipeline_result['latency_ms']:.0f}ms")
    
    # Aggregate
    summary = aggregate_results(per_question_results)
    
    return {
        "summary": summary,
        "per_question_results": per_question_results,
        "eval_set_path": str(eval_path),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def save_results(eval_output: dict) -> Path:
    """Save evaluation results to JSON file in results/ directory.
    
    Args:
        eval_output: Output from run_evaluation()
        
    Returns:
        Path to saved results file
    """
    results_dir = Path(settings.results_path)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate timestamped filename
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    results_file = results_dir / f"eval_results_{timestamp}.json"
    
    with open(results_file, "w") as f:
        json.dump(eval_output, f, indent=2)
    
    return results_file


def print_summary(eval_output: dict) -> None:
    """Print summary metrics to stdout.
    
    Args:
        eval_output: Output from run_evaluation()
    """
    summary = eval_output["summary"]
    
    print("\n" + "="*70)
    print("EVALUATION SUMMARY")
    print("="*70)
    print(f"Total queries: {summary['total_queries']}")
    print(f"Abstained: {summary['total_abstained']}/{summary['total_queries']}")
    print()
    print(f"Hit rate (retrieval): {summary['hit_rate_pct']:.2f}%")
    print(f"Abstention precision: {summary['abstention_precision'] or 'N/A'}")
    print(f"Abstention recall: {summary['abstention_recall'] or 'N/A'}")
    print(f"Mean faithfulness (non-abstained): {summary['mean_faithfulness']:.3f}")
    print()
    print(f"Mean latency: {summary['mean_latency_ms']:.2f}ms")
    print(f"P95 latency: {summary['p95_latency_ms']:.2f}ms")
    print("="*70 + "\n")


if __name__ == "__main__":
    try:
        eval_output = run_evaluation()
        results_file = save_results(eval_output)
        print_summary(eval_output)
        print(f"✓ Results saved to {results_file}\n")
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)
