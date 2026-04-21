# ⚡ ContextLite

**Post-RAG context optimizer.** Takes the chunks your RAG system already retrieved and compresses them into the highest-signal context before the LLM call — removing noise, deduplicating repeated content, and fitting everything into your token budget.

No LLM calls. Runs entirely locally. Single embedding pass.

---

## The Problem

RAG retrieves at the **chunk level**. That creates three problems:

1. **Intra-chunk noise** — a chunk is included even if only 2 of its 10 sentences are relevant to the query. The other 8 waste tokens.
2. **Cross-chunk redundancy** — if 3 chunks say the same thing in different words, all 3 go into the prompt. RAG retrieves chunks independently; it never compares them to each other.
3. **No token budget awareness** — RAG doesn't know you have a 2,048 token limit. It retrieves what it retrieves.

ContextLite sits in the gap between retrieval and the LLM call:

```
Your data → [RAG retrieval] → chunks → [ContextLite] → optimized context → LLM
```

---

## How It Works

Five steps, all local, one embedding pass:

| Step | What it does | Why RAG doesn't |
|---|---|---|
| **1. Clean** | Strip HTML tags, markdown headers, nav text, copyright boilerplate | RAG doesn't parse noise |
| **2. Score** | Embed every sentence + query in one batch; drop sentences below relevance threshold | RAG works at chunk level, not sentence level |
| **3. Dedup** | Find near-identical sentences across chunks via cosine similarity; keep the best one | RAG retrieves chunks independently |
| **4. MMR** | Rerank remaining sentences to balance relevance with topic diversity | RAG ranks by relevance only — you get clusters |
| **5. Pack** | Greedily fill token budget with highest-ranked sentences; restore reading order | RAG doesn't know your budget |

**Performance:** steps 3–4 reuse the embeddings computed in step 2, so there is exactly **one model call** per optimize run.

---

## Quickstart

```bash
git clone https://github.com/VinGuar/AICompressor.git
cd AICompressor
pip install -r requirements.txt

# UI (downloads ~80MB model on first run, cached after)
python -m streamlit run app.py
```

No API keys needed. First run downloads `all-MiniLM-L6-v2` (~80MB) and caches it locally.

---

## Usage

### UI
```bash
python -m streamlit run app.py
```
Opens at `http://localhost:8501`. Click **Load demo data** to try it immediately.

The UI has three output tabs:
- **Optimized context** — the result, ready to copy into a prompt
- **Before / After** — side-by-side comparison with token counts
- **What was removed** — colour-coded kept (green) vs removed (red) sentences

### CLI
```bash
# Pass chunks directly
python main.py --chunks "chunk one..." "chunk two..." --query "your question"

# Load from file (blank line = chunk separator)
python main.py --file chunks.txt --query "your question" --budget 1024

# Built-in demo
python main.py --demo

# Raw JSON output
python main.py --chunks "..." --query "..." --json-out
```

### API
```bash
uvicorn api:app --reload
# Docs at http://localhost:8000/docs
```

```json
POST /optimize
{
  "chunks": ["chunk one...", "chunk two..."],
  "query": "What are the pricing plans?",
  "token_budget": 512,
  "relevance_threshold": 0.25,
  "dedup_threshold": 0.85,
  "mmr_lambda": 0.7
}
```

---

## Settings

| Parameter | Default | Effect |
|---|---|---|
| `token_budget` | `2048` | Max tokens in output. Lower = more compression. Raise if output cuts off useful content. |
| `relevance_threshold` | `0.25` | Min cosine similarity for a sentence to survive step 2. `0.15` = loose, `0.4` = strict. |
| `dedup_threshold` | `0.85` | Similarity above which two sentences are considered duplicates. `0.95` = only exact matches, `0.7` = aggressive. |
| `mmr_lambda` | `0.7` | `1.0` = rank purely by relevance. `0.5` = balance relevance with covering different subtopics. |

---

## Architecture

```
contextlite/
├── cleaner.py     # Regex-based boilerplate removal (HTML, markdown, nav text)
├── embedder.py    # sentence-transformers wrapper — all-MiniLM-L6-v2, cached singleton
├── scorer.py      # Sentence splitting + single-batch embedding + relevance scoring
│                  # Stores embedding on each ScoredSentence for downstream reuse
├── deduper.py     # Cross-chunk dedup using stored embeddings (no model call)
├── mmr.py         # MMR reranking using stored embeddings + query_vec (no model call)
├── packer.py      # Greedy token budget packing via tiktoken; restores reading order
└── pipeline.py    # Orchestrates all 5 steps; single public function: optimize()
app.py             # Streamlit UI — model cached with @st.cache_resource,
│                  # results cached with @st.cache_data (identical inputs = instant)
main.py            # CLI entrypoint
api.py             # FastAPI server
```

**Key design decisions:**
- Embeddings are computed once in `scorer.py` and stored on `ScoredSentence.embedding`. This means `deduper.py` and `mmr.py` do zero model calls — they reuse what scorer already computed.
- The Streamlit model cache (`@st.cache_resource`) means the 80MB model loads once per server lifetime, not once per request.
- `@st.cache_data` on `run_optimize` means repeated clicks with the same inputs return instantly.

---

## Requirements

- Python 3.11+
- ~80MB disk for embedding model (downloaded on first run via HuggingFace)
- No GPU needed — CPU inference is fast enough for typical RAG chunk sizes
- No API keys

```
sentence-transformers  # local embeddings (all-MiniLM-L6-v2)
tiktoken               # accurate token counting
numpy <2.0             # pinned to avoid binary incompatibility with system pandas/sklearn
fastapi + uvicorn      # API server
streamlit              # UI
```
