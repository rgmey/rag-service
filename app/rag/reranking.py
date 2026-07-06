# app/rag/reranking.py
"""
LLM-based re-ranking.

Vector search retrieves by embedding similarity, which is a decent but
imperfect proxy for "is this chunk actually useful for answering the
question" — a chunk can share vocabulary with the query while not
actually answering it. Re-ranking asks the LLM itself to score each
retrieved candidate's relevance, then keeps only the best ones.

This trades one extra (cheap, low-token) LLM call for materially better
context quality, since the caller typically over-fetches (e.g. 12
candidates for a final top-4) before handing results here.
"""
import json
import logging

from openai import APIConnectionError, APIStatusError, OpenAI, RateLimitError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.core.config import settings

logger = logging.getLogger(__name__)

client = OpenAI(
    api_key=settings.OPENROUTER_API_KEY,
    base_url=settings.OPENROUTER_BASE_URL,
)

_RETRYABLE = (RateLimitError, APIConnectionError, APIStatusError)

_RERANK_PROMPT = (
    "You are scoring how relevant each passage is to a question, on a "
    "scale from 0 (irrelevant) to 10 (directly answers it).\n\n"
    "Question: {question}\n\n"
    "Passages:\n{passages}\n\n"
    "Respond with ONLY a JSON array of {n} integers, one score per "
    "passage, in the same order as listed and nothing else. "
    "Example: [7, 2, 9]"
)


@retry(
    retry=retry_if_exception_type(_RETRYABLE),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    reraise=True,
)
def _score_passages(question: str, passages: list[str]) -> list[int]:
    passage_block = "\n".join(f"[{i}] {p}" for i, p in enumerate(passages))
    prompt = _RERANK_PROMPT.format(question=question, passages=passage_block, n=len(passages))

    response = client.chat.completions.create(
        model=settings.RERANK_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )

    raw = response.choices[0].message.content.strip()
    return json.loads(raw)


def rerank(question: str, candidates: list[str], top_k: int) -> list[str]:
    """
    Re-scores `candidates` for relevance to `question` and returns the
    best `top_k`, best first.

    Falls back to the original (vector-search) order, truncated to
    `top_k`, if scoring fails or returns something unusable — a flaky
    rerank call should degrade the result, not break the chat endpoint.
    """
    if not candidates:
        return []

    if len(candidates) <= top_k:
        return candidates

    try:
        scores = _score_passages(question, candidates)
        if len(scores) != len(candidates):
            raise ValueError("score count did not match candidate count")
    except Exception:
        logger.warning("Re-ranking failed, falling back to vector-search order", exc_info=True)
        return candidates[:top_k]

    ranked = [
        doc
        for doc, _score in sorted(
            zip(candidates, scores, strict=True), key=lambda pair: pair[1], reverse=True
        )
    ]
    return ranked[:top_k]
