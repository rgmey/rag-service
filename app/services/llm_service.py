# app/services/llm_service.py
import logging

from openai import APIConnectionError, APIStatusError, OpenAI, RateLimitError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.core.config import settings

logger = logging.getLogger(__name__)

client = OpenAI(
    api_key=settings.OPENROUTER_API_KEY,
    base_url=settings.OPENROUTER_BASE_URL,
)

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the user's question using only "
    "the provided context. If the context doesn't contain the answer, "
    "say you don't know rather than guessing."
)

_CONDENSE_PROMPT = (
    "Given this conversation history and a follow-up question, rewrite the "
    "follow-up as a standalone question that makes sense without the "
    "history. If it's already standalone, return it unchanged. Respond "
    "with ONLY the rewritten question, nothing else.\n\n"
    "History:\n{history}\n\nFollow-up question: {question}"
)

_RETRYABLE = (RateLimitError, APIConnectionError, APIStatusError)


@retry(
    retry=retry_if_exception_type(_RETRYABLE),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    reraise=True,
)
def ask_llm(
    question: str,
    context: list[str] | None = None,
    history: list[dict] | None = None,
    model: str | None = None,
) -> str:
    """
    Answers `question` grounded in `context` chunks. `history` (prior
    {"role", "content"} turns from this session) is passed in as earlier
    messages so the model can resolve references like "and last year?" —
    only the current turn carries the retrieved context, so older context
    doesn't pile up in the prompt turn after turn.
    """
    model = model or settings.DEFAULT_MODEL

    if context:
        context_block = "\n\n---\n\n".join(context)
        user_content = f"Context:\n{context_block}\n\nQuestion: {question}"
    else:
        user_content = question

    messages = [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}]
    messages.extend(history or [])
    messages.append({"role": "user", "content": user_content})

    response = client.chat.completions.create(
        model=model,
        messages=messages,
    )

    return response.choices[0].message.content


def condense_question(question: str, history: list[dict]) -> str:
    """
    Rewrites a follow-up question into a standalone one using chat
    history, so retrieval isn't thrown off by pronouns/context like
    "what about it?" — embedding that literally would retrieve near
    nothing useful.

    Best-effort: falls back to the original question if this call fails,
    since a broken rewrite shouldn't break the whole chat turn.
    """
    if not history:
        return question

    history_block = "\n".join(f"{m['role']}: {m['content']}" for m in history)
    prompt = _CONDENSE_PROMPT.format(history=history_block, question=question)

    try:
        response = client.chat.completions.create(
            model=settings.DEFAULT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return response.choices[0].message.content.strip()
    except Exception:
        logger.warning("Question condensing failed, using original question", exc_info=True)
        return question
