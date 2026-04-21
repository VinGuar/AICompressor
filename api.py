"""
ContextLite FastAPI server.

Start: uvicorn api:app --reload
Docs:  http://localhost:8000/docs
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(
    title="ContextLite API",
    description=(
        "Post-RAG context optimizer. Pass your retrieved chunks and query — "
        "get back compressed, deduplicated, diversity-reranked context "
        "fitted to your token budget. No LLM calls."
    ),
    version="0.2.0",
)


class OptimizeRequest(BaseModel):
    chunks: list[str] = Field(..., description="Retrieved chunks from your RAG system")
    query: str = Field(..., description="The user query these chunks will answer")
    token_budget: int = Field(default=2048, ge=64, le=32768, description="Max output tokens")
    relevance_threshold: float = Field(default=0.25, ge=0.0, le=1.0)
    dedup_threshold: float = Field(default=0.85, ge=0.5, le=1.0)
    mmr_lambda: float = Field(default=0.7, ge=0.0, le=1.0,
                              description="1.0=pure relevance, 0.0=pure diversity")


class OptimizeResponse(BaseModel):
    optimized_context: str
    kept_sentences: list[str]
    removed_sentences: list[str]
    token_estimate_before: int
    token_estimate_after: int
    compression_ratio: float
    explanation: list[str]


@app.get("/")
def root():
    return {
        "name": "ContextLite",
        "version": "0.2.0",
        "description": "Post-RAG context optimizer — no LLM calls",
        "docs": "/docs",
        "endpoint": "POST /optimize",
    }


@app.post("/optimize", response_model=OptimizeResponse)
def optimize_context(req: OptimizeRequest):
    if not req.chunks:
        raise HTTPException(status_code=400, detail="chunks list cannot be empty")
    try:
        from contextlite.pipeline import optimize
        return optimize(
            chunks=req.chunks,
            query=req.query,
            token_budget=req.token_budget,
            relevance_threshold=req.relevance_threshold,
            dedup_threshold=req.dedup_threshold,
            mmr_lambda=req.mmr_lambda,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    return {"status": "ok"}
