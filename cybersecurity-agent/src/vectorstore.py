"""ChromaDB vector store for the cybersecurity-agent RAG pipeline."""
from __future__ import annotations

import chromadb
from sentence_transformers import SentenceTransformer

_EMBED_MODEL = "all-MiniLM-L6-v2"


def build_vectorstore(
    chunks: list[str],
    collection_name: str = "security_standards",
    persist_dir: str | None = None,
) -> chromadb.Collection:
    """Embed text chunks and store them in a ChromaDB collection.

    Args:
        chunks: List of text chunks to embed and store.
        collection_name: Name of the ChromaDB collection.
        persist_dir: Optional directory for persistent storage.
                     If None, uses an in-memory (ephemeral) client.

    Returns:
        The populated ChromaDB collection.
    """
    model = SentenceTransformer(_EMBED_MODEL)
    embeddings = model.encode(chunks)

    if persist_dir:
        client = chromadb.PersistentClient(path=persist_dir)
    else:
        client = chromadb.Client()

    collection = client.get_or_create_collection(name=collection_name)
    collection.add(
        documents=chunks,
        embeddings=embeddings.tolist(),
        ids=[f"chunk_{i}" for i in range(len(chunks))],
    )
    return collection


def query_vectorstore(
    collection: chromadb.Collection,
    query: str,
    k: int = 5,
) -> list[str]:
    """Retrieve the top-k most relevant chunks for a query.

    Args:
        collection: ChromaDB collection to search.
        query: Natural language query string.
        k: Number of results to return.

    Returns:
        List of matching document strings.
    """
    model = SentenceTransformer(_EMBED_MODEL)
    query_embedding = model.encode([query])

    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=k,
    )
    return results["documents"][0]
