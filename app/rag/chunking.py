# app/rag/chunking.py
import logging

logger = logging.getLogger(__name__)

try:
    import tiktoken

    _ENCODER = tiktoken.get_encoding("cl100k_base")
except Exception:
    # Broad on purpose: tiktoken can fail not just on missing import but
    # also on network errors fetching its BPE file (offline/air-gapped
    # environments, sandboxed CI, etc). Falling back to character-based
    # chunking keeps the service usable either way.
    logger.warning("tiktoken encoder unavailable, falling back to character-based chunking")
    _ENCODER = None


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> list[str]:
    """
    Splits text into overlapping chunks. Token-based (matches how the
    embedding model actually sees the text) when tiktoken is available,
    falls back to character-based otherwise.
    """
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    if not text:
        return []

    if _ENCODER is not None:
        tokens = _ENCODER.encode(text)
        chunks = []
        start = 0
        while start < len(tokens):
            end = start + chunk_size
            chunks.append(_ENCODER.decode(tokens[start:end]))
            start = end - overlap
        return chunks

    # Fallback: character-based chunking
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks
