# Local RAG Evaluation Harness

## Overview

A reference-based evaluation system for retrieval-augmented generation (RAG) that runs entirely offline on CPU. The project treats evaluation as the primary artefact: the RAG pipeline exists to create a concrete system to measure. The design philosophy is *fail closed* — abstain rather than fabricate when the corpus cannot support an answer.

Runs on Intel i7 with 16 GB RAM using Ollama (local LLM inference), sentence-transformers (local embeddings), and Chroma (persistent vector store). No API keys, no data egress.

## Architecture

```
INGESTION
  Document corpus (raw text) → Character-based chunking (500 chars, 50-char overlap)
  → all-MiniLM-L6-v2 embeddings → Chroma vector store (cosine distance)

RETRIEVAL & GENERATION
  User query → Embed → Retrieve top-k candidates
    → Hard abstention gate (distance threshold) → LLM (llama3.2:3b)
    → Soft abstention detection (phrase matching) → Answer

EVALUATION
  Reference-based metrics (retrieval hit, abstention correctness, substring faithfulness)
  → Aggregate statistics over labelled eval set → JSON results
```

## Key Design Decisions

**1. Reference-based metrics over LLM-as-judge**

Metric functions check retrieval against ground-truth document IDs, abstention decisions against binary labels, and answer text for required substrings. Trade-off: substring matching is coarser than semantic similarity (a judge would score partial correctness); but the approach is deterministic, reproducible, and fast on CPU with zero judge cost or bias.

**2. Explicit abstention gate to prevent hallucination**

Two-tier abstention: hard gate refuses before calling the LLM when best retrieval distance exceeds a threshold; soft tier lets the LLM read topically-relevant context and decline because the specific fact is absent. Both count as abstention when scored. Trade-off: system errs towards refusal rather than speculation, which is conservative; on small corpora, abstention recall is imperfect because LLM refusals are detected via pattern matching on answer text, not a structured signal.

**3. Provider abstraction for multi-backend support**

LLMProvider abstract base class with OllamaProvider implementation. Generate and retrieve logic does not import provider directly; swapping to a hosted model requires only a new provider implementation and a config change. Trade-off: abstraction adds minimal scaffolding; the pattern scales if adding multiple backends.

**4. Fully local pipeline**

No egress, no rate limits, no external dependencies. Reproducible evaluation on the same hardware. Trade-off: constrained to single-machine resources; latency on CPU is high (9.4 s mean, 25.3 s p95 per query).

## Results

Measured on Intel i7-1165G7, CPU only, llama3.2:3b, 7-document synthetic corpus, 30-question eval set (22 answerable, 8 unanswerable; 4 answerable questions require multiple documents).

| Metric | Value |
|--------|-------|
| Retrieval hit-rate (answerable only) | 95.5% |
| Abstention precision | 0.78 |
| Abstention recall | 0.88 |
| Faithfulness (non-abstained answers) | 0.93 |
| Mean latency | 9.4 s |
| P95 latency | 25.3 s |

The corpus is synthetic and deliberately so: ground truth is unambiguous and evaluation is not conflated with source quality.

## Where the System Fails

**Retrieval misses on multi-hop questions.**
One question required facts from three separate documents; the system retrieved only two and abstained rather than guessing. This is correct behaviour (fails closed), but it illustrates single-shot retrieval's weakness on decomposable queries.

**Abstention detection is a heuristic.**
Abstention is scored by pattern matching: the system searches the LLM's answer text for 18 refusal phrases (`"I don't know"`, `"not enough information"`, etc.). This introduces residual false positives (low precision: 0.78) and false negatives (incomplete recall: 0.88). The honest fix is a structured abstention signal from the generator (a flag or token), not a longer phrase list, which would overfit the metric.

**Small-model degradation.**
One non-abstained answer exhibited a faithfulness failure where the 3B model truncated a title and then incorrectly claimed the untruncated version was absent from the corpus. This is a small-model-on-CPU limitation, not a pipeline fault; a larger model or quantisation strategy would likely improve this.

## Limitations

- **Coarse faithfulness metric:** substring matching does not detect semantic drift or hallucination of plausible false details.
- **Heuristic abstention detection:** phrase matching has residual false positives and negatives; no structured signal from the generator.
- **Single-pass retrieval:** weak on multi-hop queries that require query decomposition or iterative retrieval.
- **CPU-only performance:** mean latency is 9.4 s per query; p95 exceeds 25 s.
- **Character-based chunking:** deliberately simple; no semantic awareness of chunk boundaries.
- **Synthetic corpus:** unambiguous but not representative of real-world document noise or scale.

## Future Work

- Structured abstention output from the generator (a flag or explicit refusal token) to replace phrase matching.
- Semantic faithfulness scoring (BLEU, ROUGE, or BERTScore) as a coarser baseline or cross-check.
- Query decomposition and iterative retrieval for multi-hop reasoning.
- Semantic chunking to replace character-based splitting.
- Optional LLM-as-judge variant over a small subset of questions as a baseline.
- Multilingual retrieval variant (cross-lingual embeddings).

## How to Run

**Setup:**
```bash
python -m venv venv
. venv/Scripts/activate  # Windows: venv\Scripts\activate
pip install -e .
```

**Ingest corpus:**
```bash
ollama pull llama3.2:3b
python -m src.ingest
```

**Query interactively:**
```bash
python -m src.pipeline "Who was Margriet Dekker?"
```

**Run full evaluation:**
```bash
python -m src.evals.run_eval
```

Evaluation output is printed to stdout and saved as a timestamped JSON file to `results/`.

## Project Structure

```
src/
  config.py           — Pydantic settings (model names, paths, thresholds)
  providers.py        — LLMProvider abstract class, OllamaProvider impl
  ingest.py           — Chunk, embed, store in Chroma
  retrieve.py         — Query → cosine similarity search → top-k
  generate.py         — Retrieved chunks + LLM → answer, abstention flag
  pipeline.py         — Orchestration: retrieve + generate + latency
  evals/
    metrics.py        — Pure functions: hit rate, abstention correctness, faithfulness
    run_eval.py       — Load eval set, run pipeline, aggregate results

tests/
  test_metrics.py     — 35 unit tests (all passing)

data/
  eval_set.jsonl      — 30 labelled questions with ground truth
  corpus/             — 7 synthetic documents

results/
  eval_results_*.json — Timestamped evaluation output
```

## Dependencies

- Python 3.11+
- Ollama (local LLM daemon)
- chromadb 0.4.x
- sentence-transformers (all-MiniLM-L6-v2)
- pydantic v2+
- pytest (dev)
