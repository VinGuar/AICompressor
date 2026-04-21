from .cleaner import clean
from .scorer import score_sentences
from .deduper import deduplicate
from .mmr import mmr_rerank
from .packer import pack, count_tokens


def optimize(
    chunks: list[str],
    query: str,
    token_budget: int = 2048,
    relevance_threshold: float = 0.25,
    dedup_threshold: float = 0.85,
    mmr_lambda: float = 0.7,
) -> dict:
    """
    Post-RAG context optimization. Single embedding pass — no LLM calls.

    Pipeline:
      1. Clean  — strip boilerplate, HTML, markdown noise from each chunk
      2. Score  — embed query + all sentences once; keep sentences above threshold
      3. Dedup  — drop near-identical sentences across chunks (uses stored embeddings)
      4. MMR    — rerank for relevance + diversity (uses stored embeddings)
      5. Pack   — greedily fill token budget; restore original reading order

    Args:
        chunks:               Retrieved chunks from your RAG system
        query:                The user query these chunks will answer
        token_budget:         Max tokens the output context may use
        relevance_threshold:  Min cosine sim [0–1] for a sentence to survive step 2
        dedup_threshold:      Cosine sim [0–1] above which two sentences are duplicates
        mmr_lambda:           MMR balance — 1.0 = pure relevance, 0.0 = pure diversity
    """
    raw_text = "\n\n".join(chunks)
    tokens_before = count_tokens(raw_text)

    # Step 1: clean
    cleaned = [clean(c) for c in chunks]
    cleaned = [c for c in cleaned if len(c.strip()) > 20]

    # Step 2: score — ONE embed call for query + all sentences
    scored, total_sentences, query_vec = score_sentences(
        cleaned, query, threshold=relevance_threshold
    )
    n_low_relevance = total_sentences - len(scored)

    # Step 3: dedup — uses embeddings already stored on each ScoredSentence
    deduped, removed_dups = deduplicate(scored, sim_threshold=dedup_threshold)

    # Step 4: MMR — uses stored embeddings + query_vec from step 2
    reranked = mmr_rerank(deduped, query_vec, lambda_param=mmr_lambda)

    # Step 5: pack into token budget, restores reading order internally
    packed, over_budget = pack(reranked, token_budget)

    optimized_context = " ".join(s.text for s in packed)
    tokens_after = count_tokens(optimized_context)
    compression_ratio = round(1 - tokens_after / tokens_before, 3) if tokens_before else 0

    return {
        "optimized_context": optimized_context,
        "kept_sentences": [s.text for s in packed],
        "removed_sentences": [s.text for s in removed_dups] + [s.text for s in over_budget],
        "token_estimate_before": tokens_before,
        "token_estimate_after": tokens_after,
        "compression_ratio": compression_ratio,
        "explanation": _explain(
            n_low_relevance, len(removed_dups), len(over_budget),
            token_budget, tokens_before, tokens_after,
        ),
    }


def _explain(n_low_rel, n_deduped, n_over_budget, budget, before, after) -> list[str]:
    lines = []
    if n_low_rel:
        lines.append(f"Filtered {n_low_rel} low-relevance sentences via embedding similarity")
    if n_deduped:
        lines.append(f"Removed {n_deduped} near-duplicate sentences across chunks")
    if n_over_budget:
        lines.append(f"Left {n_over_budget} lower-priority sentences out to fit {budget:,} token budget")
    pct = round((1 - after / before) * 100, 1) if before else 0
    lines.append(f"{before:,} → {after:,} tokens  ({pct}% reduction, {before - after:,} saved)")
    return lines
