# ContextLite

```
Input (RAG output):   4,800 tokens
Output (ContextLite): 1,400 tokens
Reduction:            71%
LLM answer quality:   unchanged
```

---

RAG retrieves relevant chunks — but it still sends redundant, verbose, and overlapping text to the model. You get 8 chunks where 3 say the same thing, half the sentences in each chunk are irrelevant to the actual query, and none of it fits cleanly into your token budget. You end up paying for noise and getting worse reasoning.

ContextLite sits between your retrieval step and the LLM call. It takes what RAG returned, strips it down to only the sentences that matter, removes duplicates across chunks, reranks for coverage, and packs the best content into whatever token budget you set. No LLM calls — runs locally.

```
[your data] → [RAG retrieval] → [ContextLite] → [LLM]
                                      ↑
                       cleans, deduplicates, reranks,
                       fits your token budget
```

This is not RAG. RAG decides what to retrieve. ContextLite decides how to clean and pack what was retrieved — a layer that doesn't exist anywhere in LangChain or LlamaIndex.

---

## Getting started

```bash
git clone https://github.com/VinGuar/AICompressor.git
cd AICompressor
pip install -r requirements.txt
python -m streamlit run app.py
```

Downloads the embedding model (~80MB) on first run, cached after. No API keys.

---

## Usage

**UI** — paste your chunks, enter a query, see the results with before/after token counts and a breakdown of what was removed and why.

**CLI**
```bash
python main.py --chunks "chunk one" "chunk two" --query "your question"
python main.py --demo
python main.py --file chunks.txt --query "your question" --budget 1024
```

**API**
```bash
uvicorn api:app --reload
```
```json
POST /optimize
{
  "chunks": ["..."],
  "query": "What are the pricing plans?",
  "token_budget": 512
}
```
Swagger docs at `http://localhost:8000/docs`.

---

## How it works

Five steps, all local, single embedding pass:

1. **Clean** — strip HTML, markdown headers, nav text, boilerplate
2. **Score** — embed every sentence and the query together in one batch, drop anything below the relevance threshold
3. **Dedup** — find near-identical sentences across chunks, keep the highest-scoring one
4. **MMR rerank** — reorder by relevance + diversity so you don't end up with 5 sentences all saying the same thing
5. **Pack** — fill the token budget greedily with the best sentences, restore reading order

Steps 3 and 4 reuse embeddings from step 2 — there's one model call total per run.

---

## Settings

| | default | |
|---|---|---|
| `token_budget` | 2048 | hard cap on output tokens |
| `relevance_threshold` | 0.25 | raise for stricter filtering, lower to keep more |
| `dedup_threshold` | 0.85 | cosine similarity above which sentences are considered duplicates |
| `mmr_lambda` | 0.7 | 1.0 = pure relevance, 0.0 = pure diversity |

---

## Code

```
contextlite/
├── cleaner.py    # regex boilerplate removal
├── embedder.py   # sentence-transformers wrapper, cached singleton
├── scorer.py     # sentence splitting + single-batch embed + relevance scoring
├── deduper.py    # cross-chunk dedup using embeddings from scorer
├── mmr.py        # MMR reranking using same embeddings
├── packer.py     # token budget packing with tiktoken
└── pipeline.py   # orchestrates all steps, one function: optimize()
app.py            # Streamlit UI
main.py           # CLI
api.py            # FastAPI
```

Embeddings are computed once in `scorer.py` and stored on each sentence object. `deduper.py` and `mmr.py` read those directly — no extra model calls. The Streamlit app caches the model on startup so the first optimize is fast and repeated runs with the same input are instant.

Requires Python 3.11+, no GPU needed.
