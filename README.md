# ContextLite

Most RAG pipelines retrieve relevant chunks and pass them directly to the LLM. The problem is that chunks are noisy — a chunk gets included even if only 2 of its 10 sentences are relevant. Multiple chunks often repeat the same information. None of it fits neatly into a token budget. You end up paying for noise and getting worse reasoning.

ContextLite is a post-retrieval optimization layer. It takes what your RAG system returned and compresses it down to only the sentences that matter — deduplicating across chunks, reranking for coverage, and packing into a token budget — before anything reaches the LLM.

**No LLM calls. Runs locally. Plugs into any existing RAG stack.**

---

## Before / After

```
Query: "What are the pricing plans and API rate limits?"

--- INPUT (raw RAG output) ---
248 tokens across 5 chunks.
Chunk 1: pricing info + irrelevant company history
Chunk 2: API limits + unrelated support info
Chunk 3: near-duplicate of chunk 1 pricing info
Chunk 4: funding/team info (completely off-topic)
Chunk 5: mobile app info (irrelevant to query)

--- OUTPUT (ContextLite) ---
112 tokens
Only the pricing tiers, discount info, and API rate limits.
Duplicate pricing sentences removed. Off-topic chunks stripped.

Reduction: 54.8% — on a clean demo dataset.
Real RAG pipelines with noisier data see 60–70%+.
```

---

## Where it fits

```
[your data] → [RAG retrieval] → [ContextLite] → [LLM]
```

RAG handles retrieval. LangChain/LlamaIndex handle orchestration. ContextLite handles the layer neither of them touch — cleaning and packing retrieved context before the LLM call.

This is not a RAG replacement. It's what should happen after RAG and before inference.

---

## Why this matters

Token costs scale with context size. Sending 5 chunks where 3 say the same thing and 2 are off-topic doesn't just cost more — it actively makes the LLM's job harder. Redundant context increases hallucination risk and dilutes the signal.

Current RAG stacks don't solve this. They retrieve and pass. ContextLite is the missing step.

---

## What it does

Takes your retrieved chunks and runs five steps, all local:

1. **Clean** — strips HTML, markdown headers, nav text, boilerplate noise
2. **Score** — embeds every sentence and the query in one batch, drops anything below the relevance threshold
3. **Dedup** — finds near-identical sentences across chunks, keeps the highest-scoring one
4. **MMR rerank** — reorders by relevance + diversity so you don't get 5 sentences covering the same point
5. **Pack** — greedily fills the token budget with the best sentences, restores reading order

Single embedding pass. Steps 3 and 4 reuse embeddings from step 2, so there's exactly one model call per run.

---

## Getting started

```bash
git clone https://github.com/VinGuar/AICompressor.git
cd AICompressor
pip install -r requirements.txt

# UI
python -m streamlit run app.py

# or CLI
python main.py --demo
```

Downloads the embedding model (~80MB) on first run, cached after. No API keys.

---

## Usage

**UI** — `python -m streamlit run app.py`

Paste chunks, enter a query, hit Optimize. Shows before/after token counts, what was kept vs removed, and a side-by-side comparison.

**CLI**
```bash
python main.py --chunks "chunk one" "chunk two" --query "your question"
python main.py --file chunks.txt --query "your question" --budget 1024
python main.py --demo
```

**API**
```bash
uvicorn api:app --reload
```
```json
POST /optimize
{
  "chunks": ["chunk one...", "chunk two..."],
  "query": "What are the pricing plans?",
  "token_budget": 512
}
```
Swagger docs at `http://localhost:8000/docs`.

---

## Use cases

- RAG pipelines over large document sets
- AI agents with tool outputs that need to fit a context window
- Chatbots over documentation where chunks overlap heavily
- Any pipeline where you're hitting token limits or paying too much on inference

---

## Settings

| | default | |
|---|---|---|
| `token_budget` | 2048 | hard cap on output tokens |
| `relevance_threshold` | 0.25 | min similarity for a sentence to survive — raise it for stricter filtering |
| `dedup_threshold` | 0.85 | cosine similarity above which two sentences are considered duplicates |
| `mmr_lambda` | 0.7 | 1.0 = pure relevance ranking, 0.0 = pure diversity |

---

## Code structure

```
contextlite/
├── cleaner.py    # regex boilerplate removal
├── embedder.py   # sentence-transformers wrapper, cached singleton
├── scorer.py     # sentence splitting, single-batch embed, relevance scoring
├── deduper.py    # cross-chunk dedup using embeddings stored in scorer
├── mmr.py        # MMR reranking using same stored embeddings
├── packer.py     # token budget packing with tiktoken
└── pipeline.py   # orchestrates all steps — one function: optimize()
app.py            # Streamlit UI
main.py           # CLI
api.py            # FastAPI
```

Embeddings are computed once in `scorer.py` and attached to each sentence object. `deduper.py` and `mmr.py` read those directly — no extra model calls. The Streamlit app caches the model on startup so repeated runs with the same input are instant.

Requires Python 3.11+, no GPU needed.
