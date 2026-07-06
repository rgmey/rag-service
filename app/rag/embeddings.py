# app/rag/embeddings.py
from openai import APIConnectionError, APIStatusError, OpenAI, RateLimitError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.core.config import settings

client = OpenAI(
    api_key=settings.OPENROUTER_API_KEY,
    base_url=settings.OPENROUTER_BASE_URL,
)

_RETRYABLE = (RateLimitError, APIConnectionError, APIStatusError)


@retry(
    retry=retry_if_exception_type(_RETRYABLE),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    reraise=True,
)
def get_embeddings(texts: list[str]) -> list[list[float]]:
    """Batch-embed a list of chunks in a single API call instead of one call per chunk.
    Retries transient failures (rate limits, connection errors) with backoff."""
    if not texts:
        return []

    response = client.embeddings.create(
        model=settings.EMBEDDING_MODEL,
        input=texts,
    )

    return [item.embedding for item in response.data]


def get_embedding(text: str) -> list[float]:
    """Single-text convenience wrapper (e.g. for embedding a user query)."""
    return get_embeddings([text])[0]
