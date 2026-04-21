import tiktoken
from .scorer import ScoredSentence

_enc = None


def _get_encoder():
    global _enc
    if _enc is None:
        try:
            _enc = tiktoken.get_encoding("cl100k_base")
        except Exception:
            pass
    return _enc


def count_tokens(text: str) -> int:
    enc = _get_encoder()
    if enc:
        return len(enc.encode(text))
    return max(1, len(text) // 4)


def pack(
    sentences: list[ScoredSentence],
    token_budget: int,
) -> tuple[list[ScoredSentence], list[ScoredSentence]]:
    """
    Greedily pack sentences into token_budget, highest MMR score first.
    Returns (packed, over_budget).
    """
    packed: list[ScoredSentence] = []
    over_budget: list[ScoredSentence] = []
    used = 0

    for s in sentences:
        t = count_tokens(s.text)
        if used + t <= token_budget:
            packed.append(s)
            used += t
        else:
            over_budget.append(s)

    # Restore reading order: sort by (chunk_idx, sent_idx)
    packed.sort(key=lambda s: (s.chunk_idx, s.sent_idx))
    return packed, over_budget
