# ContextLite

RAG retrieves chunks. The problem is it retrieves whole chunks — even if only 2 sentences in a chunk are relevant, all 10 go into your prompt. And if 3 different chunks say the same thing, all 3 go in. By the time it hits the LLM you're paying for a lot of noise.

ContextLite sits between your retrieval step and the LLM call. It takes the chunks you already retrieved, strips out the irrelevant sentences, removes duplicates across chunks, reranks for diversity, and packs everything into a token budget. No LLM calls — runs locally with a small embedding model.

```
RAG chunks → ContextLite → clean context → LLM
```

## Getting started

```bash
git clone https://github.com/VinGuar/AICompressor.git
cd AICompressor
pip install -r requirements.txt
python -m streamlit run app.py
```

First run downloads the embedding model (~80MB, cached after). No API keys needed.

## How to use it

**UI** — `python -m streamlit run app.py`

Paste your chunks, enter your query, hit Optimize. Shows before/after token counts, what was kept vs removed, and a side-by-side comparison.

**CLI**
```bash
python main.py --chunks "chunk one" "chunk two" --query "your question"
python main.py --demo   # try with built-in example data
python main.py --file chunks.txt --query "your question" --budget 1024
```

**API** — `uvicorn api:app --reload`, then `POST /optimize`
```json
{
  "chunks": ["chunk one...", "chunk two..."],
  "query": "What are the pricing plans?",
  "token_budget": 512
}
```
Docs at `http://localhost:8000/docs`.

## What it actually does

Five steps, all local:

1. **Clean** — strips HTML, markdown headers, nav text, boilerplate
2. **Score** — embeds every sentence and the query in one batch, drops sentences below the relevance threshold
3. **Dedup** — finds near-identical sentences across chunks and removes them
4. **MMR rerank** — reorders by relevance + diversity so you don't get 5 chunks all saying the same thing
5. **Pack** — greedily fills the token budget with the best sentences, restores original reading order

Steps 3 and 4 reuse the embeddings from step 2, so there's only one model call per run.

## Settings

| | default | what it does |
|---|---|---|
| `token_budget` | 2048 | hard cap on output tokens |
| `relevance_threshold` | 0.25 | how similar a sentence needs to be to the query to survive — raise it to get stricter filtering |
| `dedup_threshold` | 0.85 | how similar two sentences need to be to count as duplicates — lower it to be more aggressive |
| `mmr_lambda` | 0.7 | 1.0 = pure relevance ranking, 0.0 = pure diversity, 0.7 is a good middle ground |

## Code structure

```
contextlite/
├── cleaner.py    # regex boilerplate removal
├── embedder.py   # sentence-transformers wrapper, cached singleton
├── scorer.py     # splits sentences, embeds everything in one batch, scores against query
├── deduper.py    # cross-chunk dedup using embeddings stored in scorer
├── mmr.py        # MMR reranking using same stored embeddings
├── packer.py     # token budget packing with tiktoken
└── pipeline.py   # puts it all together, one function: optimize()
app.py            # Streamlit UI
main.py           # CLI
api.py            # FastAPI
```

The embeddings are computed once in `scorer.py` and stored on each sentence object. `deduper.py` and `mmr.py` just read those — no extra model calls. The Streamlit app caches the model on startup and caches results, so repeated runs with the same input are instant.

Requires Python 3.11+, no GPU needed.
