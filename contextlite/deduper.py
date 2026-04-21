from .scorer import ScoredSentence
from .embedder import cosine_sim


def deduplicate(
    sentences: list[ScoredSentence],
    sim_threshold: float = 0.85,
) -> tuple[list[ScoredSentence], list[ScoredSentence]]:
    """
    Remove near-duplicate sentences using pre-computed embeddings (no API/model call).

    Sentences arrive sorted by score descending — we keep the highest-scored
    version of any near-duplicate pair (i.e. first one wins).

    Returns: (kept, removed)
    """
    if not sentences:
        return [], []

    kept: list[ScoredSentence] = []
    removed: list[ScoredSentence] = []

    for sentence in sentences:
        is_dup = any(
            cosine_sim(sentence.embedding, k.embedding) >= sim_threshold
            for k in kept
        )
        if is_dup:
            removed.append(sentence)
        else:
            kept.append(sentence)

    return kept, removed
