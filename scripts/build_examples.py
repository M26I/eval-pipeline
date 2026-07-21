"""One-off script to build data/example_answers.json from real pipeline outputs.

Run once with Ollama running locally, then commit the generated JSON.
The /examples API endpoint serves this file directly with no model call.

Usage:
    python scripts/build_examples.py
"""

import json
import sys
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline import answer

# Questions chosen to demonstrate key system behaviours:
#   1. Answered question with a single source           — normal RAG response
#   2. Answered question requiring multiple sources     — multi-doc retrieval
#   3. Abstention: specific fact absent from corpus     — soft fail-closed
#   4. Abstention: out-of-domain question               — hard fail-closed
EXAMPLE_QUESTIONS = [
    {
        "question": "Who was Pieter van Loon?",
        "note": "Single-source answered question: biographical detail from corpus",
    },
    {
        "question": "Who taught the painter of Woman Reading by a North Window?",
        "note": "Multi-source answered question: requires linking painting to artist to teacher",
    },
    {
        "question": "What does the painting The Blue Kitchen depict?",
        "note": "Soft abstention: painting does not exist in the corpus; model declines rather than fabricates",
    },
    {
        "question": "What is the capital of France?",
        "note": "Hard abstention: out-of-domain question, no relevant retrieval",
    },
]

OUTPUT_PATH = Path("data/example_answers.json")


def build() -> None:
    print(f"Building {len(EXAMPLE_QUESTIONS)} examples...")
    results = []

    for item in EXAMPLE_QUESTIONS:
        question = item["question"]
        note = item["note"]
        print(f"\n  Q: {question}")

        result = answer(question)

        entry = {
            "question": question,
            "answer": result["answer"],
            "abstained": result["abstained"],
            "sources": result["sources"],
            "note": note,
        }
        results.append(entry)
        print(f"  Abstained: {result['abstained']} | Sources: {result['sources']}")
        print(f"  Latency: {result['latency_ms']}ms")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nWrote {len(results)} examples to {OUTPUT_PATH}")


if __name__ == "__main__":
    build()
