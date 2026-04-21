import re
from dataclasses import dataclass, field
import numpy as np
from .embedder import embed, batch_cosine_sim


@dataclass
class ScoredSentence:
    text: str
    score: float
    chunk_idx: int
    sent_idx: int
    embedding: np.ndarray = field(default=None, compare=False, repr=False)


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


def split_sentences(text: str) -> list[str]:
    parts = _SENTENCE_SPLIT.split(text)
    return [s.strip() for s in parts if len(s.strip()) > 25]


def score_sentences(
    chunks: list[str],
    query: str,
    threshold: float = 0.25,
) -> tuple[list[ScoredSentence], int, np.ndarray]:
    """
    Embed everything in one batch. Returns:
      (scored_sentences sorted by score desc, total_sentence_count, query_vec)

    Embeddings are stored on each ScoredSentence so downstream steps
    (deduper, mmr) never need to call embed() again.
    """
    all_sentences: list[tuple[str, int, int]] = []
    for chunk_idx, chunk in enumerate(chunks):
        for sent_idx, sentence in enumerate(split_sentences(chunk)):
            all_sentences.append((sentence, chunk_idx, sent_idx))

    if not all_sentences:
        return [], 0, np.array([])

    texts = [s for s, _, _ in all_sentences]

    # Single embed call for query + all sentences together
    all_vecs = embed([query] + texts)
    query_vec = all_vecs[0]
    sent_matrix = all_vecs[1:]

    scores = batch_cosine_sim(query_vec, sent_matrix)

    results = []
    for i, (text, chunk_idx, sent_idx) in enumerate(all_sentences):
        if scores[i] >= threshold:
            results.append(ScoredSentence(
                text=text,
                score=float(scores[i]),
                chunk_idx=chunk_idx,
                sent_idx=sent_idx,
                embedding=sent_matrix[i],
            ))

    results.sort(key=lambda s: s.score, reverse=True)
    return results, len(all_sentences), query_vec
