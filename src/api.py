"""FastAPI web layer for the RAG pipeline.

Exposes the existing pipeline over HTTP without modifying core logic.
Respects LLM_PROVIDER env var for provider selection (Ollama local or Groq hosted).
"""

import json
import logging
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field
from src.pipeline import answer
from src.evals.metrics import is_soft_abstention


class ASCIISafeJSONResponse(JSONResponse):
    """JSONResponse that escapes non-ASCII characters as \\uXXXX sequences.

    Prevents UTF-8 byte corruption when clients (e.g. PowerShell Invoke-RestMethod)
    decode the response using a non-UTF-8 default encoding.
    """

    def render(self, content) -> bytes:
        return json.dumps(
            content,
            ensure_ascii=True,
            allow_nan=False,
            indent=None,
            separators=(",", ":"),
        ).encode("ascii")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models for request/response validation
class AskRequest(BaseModel):
    """Request body for /ask endpoint."""
    question: str = Field(..., min_length=1, description="User question to answer")
    
    model_config = {
        "json_schema_extra": {
            "example": {"question": "Who was Margriet Dekker?"}
        }
    }


class AskResponse(BaseModel):
    """Response body for /ask endpoint."""
    answer: str = Field(..., description="Generated answer from pipeline")
    abstained: bool = Field(..., description="Whether system abstained (hard or soft)")
    sources: list[str] = Field(default_factory=list, description="Document IDs used as sources")
    latency_ms: float = Field(..., description="Total pipeline latency in milliseconds")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "answer": "Margriet Dekker was a fictional painter...",
                "abstained": False,
                "sources": ["margriet_dekker"],
                "latency_ms": 9385.81
            }
        }
    }


class HealthResponse(BaseModel):
    """Response body for /health endpoint."""
    status: str = Field(..., description="Service status")


class ExampleEntry(BaseModel):
    """A single cached example result from data/example_answers.json."""
    question: str = Field(..., description="The question that was asked")
    answer: str = Field(..., description="The pipeline's answer")
    abstained: bool = Field(..., description="Whether the system abstained")
    sources: list[str] = Field(default_factory=list, description="Source doc_ids used")
    note: str = Field(default="", description="Human-readable note about this example")


# Path to the pre-computed examples file
_EXAMPLES_PATH = Path("data/example_answers.json")

# Path to the landing page HTML
_INDEX_PATH = Path("src/static/index.html")


# Create FastAPI app
app = FastAPI(
    title="Local RAG Pipeline API",
    description="FastAPI wrapper for offline RAG pipeline with Ollama or Groq provider",
    version="0.1.0",
    default_response_class=ASCIISafeJSONResponse,
)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Health check endpoint for uptime monitoring."""
    return HealthResponse(status="ok")


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def index() -> HTMLResponse:
    """Serve the landing page."""
    return HTMLResponse(content=_INDEX_PATH.read_text(encoding="utf-8"))


@app.get("/examples", response_model=list[ExampleEntry])
async def examples() -> list[ExampleEntry]:
    """Return pre-computed cached example results from data/example_answers.json.

    These are genuine pipeline outputs generated once via scripts/build_examples.py
    and committed to the repository. No model call is made at request time.

    Raises:
        HTTPException: 404 if example_answers.json has not been generated yet
        HTTPException: 500 if the file is present but cannot be parsed
    """
    if not _EXAMPLES_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail={
                "error": "example_answers.json not found",
                "hint": "Run: python scripts/build_examples.py to generate it",
            },
        )
    try:
        raw = json.loads(_EXAMPLES_PATH.read_text(encoding="utf-8"))
        return [ExampleEntry(**entry) for entry in raw]
    except Exception as e:
        logger.error(f"Failed to load examples: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={"error": str(e), "type": type(e).__name__},
        )


@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest) -> AskResponse:
    """Answer a question using the RAG pipeline.
    
    Retrieves relevant documents and generates an answer.
    Respects LLM_PROVIDER env var for provider selection.
    
    Args:
        request: AskRequest with question
        
    Returns:
        AskResponse with answer, abstention status, sources, and latency
        
    Raises:
        HTTPException: 500 if pipeline.answer() raises an exception
    """
    try:
        logger.info(f"Processing question: {request.question[:100]}...")
        
        # Call existing pipeline
        result = answer(request.question)

        # Apply soft abstention detection — same logic as run_eval.py.
        # pipeline.answer() only sets abstained=True for the hard distance gate;
        # generation-stage refusals are caught here via phrase matching.
        effective_abstain = result["abstained"] or is_soft_abstention(result["answer"])
        
        # Extract fields for response (omit retrieved_chunks, which is internal)
        response = AskResponse(
            answer=result["answer"],
            abstained=effective_abstain,
            sources=result["sources"],
            latency_ms=result["latency_ms"],
        )
        
        logger.info(
            f"Question answered. Abstained: {response.abstained}, "
            f"Latency: {response.latency_ms}ms"
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Pipeline error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={"error": str(e), "type": type(e).__name__},
        )


if __name__ == "__main__":
    import uvicorn
    
    print("Starting Local RAG Pipeline API...")
    print("Available at: http://localhost:8000")
    print("API docs at: http://localhost:8000/docs")
    print("ReDoc docs at: http://localhost:8000/redoc")
    
    uvicorn.run(
        "src.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
