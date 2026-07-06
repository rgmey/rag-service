# app/rag/retrieval.py
from app.rag.index import collection


def search(query_embedding, k: int = 3):
    """Returns top-k (document_text, distance) pairs from the shared collection."""
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
    )

    documents = results["documents"][0] if results["documents"] else []
    distances = results["distances"][0] if results.get("distances") else [None] * len(documents)

    return list(zip(documents, distances, strict=True))
