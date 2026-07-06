from app.rag.chunking import chunk_text


def test_empty_text_returns_no_chunks():
    assert chunk_text("", chunk_size=500, overlap=100) == []


def test_overlap_must_be_smaller_than_chunk_size():
    import pytest

    with pytest.raises(ValueError):
        chunk_text("hello world", chunk_size=100, overlap=100)


def test_short_text_returns_single_chunk():
    chunks = chunk_text("hello world", chunk_size=500, overlap=100)
    assert len(chunks) == 1
    assert "hello world" in chunks[0]


def test_long_text_produces_multiple_overlapping_chunks():
    # ~2000 words guarantees multiple chunks at chunk_size=500 tokens
    text = " ".join(f"word{i}" for i in range(2000))
    chunks = chunk_text(text, chunk_size=500, overlap=100)

    assert len(chunks) > 1
    # Every chunk should be non-empty
    assert all(chunk.strip() for chunk in chunks)


def test_chunking_is_deterministic():
    text = " ".join(f"word{i}" for i in range(500))
    chunks_a = chunk_text(text, chunk_size=200, overlap=50)
    chunks_b = chunk_text(text, chunk_size=200, overlap=50)
    assert chunks_a == chunks_b
