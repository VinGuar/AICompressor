import numpy as np
from .scorer import ScoredSentence
from .embedder import cosine_sim


def mmr_rerank(
    sentences: list[ScoredSentence],
    query_vec: np.ndarray,
    lambda_param: float = 0.7,
) -> list[ScoredSentence]:
    """
    Maximal Marginal Relevance reranking using pre-computed embeddings (no model call).

    At each step, picks the sentence that maximises:
        λ · sim(sentence, query) - (1-λ) · max_sim(sentence, already_selected)

    λ=1.0 → pure relevance order (same as score sort).
    λ=0.0 → pure diversity (each pick is as different as possible from prior picks).
    λ=0.7 → favours relevance but actively avoids picking near-duplicate content.

    This is the key advantage over RAG: RAG ranks by relevance only, producing
    clusters of similar chunks. MMR spreads picks across distinct subtopics.
    """
    if len(sentences) <= 1:
        return sentences

    selected: list[ScoredSentence] = []
    remaining = list(sentences)

    while remaining:
        best = max(
            remaining,
            key=lambda s: (
                lambda_param * cosine_sim(s.embedding, query_vec)
                - (1 - lambda_param) * (
                    max(cosine_sim(s.embedding, sel.embedding) for sel in selected)
                    if selected else 0.0
                )
            ),
        )
        selected.append(best)
        remaining.remove(best)

    return selected
