# app/rag/index.py
"""
Single source of truth for the vector store.

Both ingestion (pipeline.py) and retrieval (retrieval.py) import the
`collection` object from this module, so they always read/write the
same persisted collection instead of silently drifting apart.
"""

import chromadb

from app.core.config import settings

_client = chromadb.PersistentClient(path=settings.CHROMA_PERSIST_DIR)
collection = _client.get_or_create_collection(settings.CHROMA_COLLECTION_NAME)


def store_chunks(vectors: list[dict]) -> None:
    """vectors: list of {"id": str, "text": str, "embedding": list[float]}"""
    if not vectors:
        return

    collection.add(
        ids=[v["id"] for v in vectors],
        documents=[v["text"] for v in vectors],
        embeddings=[v["embedding"] for v in vectors],
    )
