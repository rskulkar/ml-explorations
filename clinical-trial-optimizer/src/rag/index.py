"""Hybrid index (FAISS + BM25) module for CI RAG pipeline.

Provides:
    - HybridIndex: dataclass combining dense (FAISS) and sparse (BM25) indexes.
    - build_index: Construct and persist a hybrid index.
    - load_index: Load a persisted hybrid index.
    - save_index: Persist a hybrid index to disk.
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class HybridIndex:
    """Hybrid index combining FAISS (dense) and BM25 (sparse) retrieval."""
    faiss_index: Optional[object] = None  # faiss.IndexFlatL2
    bm25: Optional[object] = None  # BM25Okapi
    chunks: list = field(default_factory=list)
    embeddings: Optional[object] = None  # np.ndarray


def build_index(
    chunks: list,
    embeddings,  # np.ndarray
    collection_name: str = "ci_trials",
    persist_dir: str | None = None,
) -> HybridIndex:
    """
    Build a hybrid index (FAISS + BM25) from chunks and embeddings.

    Args:
        chunks: List of CriterionChunk objects.
        embeddings: numpy array of dense embeddings.
        collection_name: Name for the index collection.
        persist_dir: Optional directory to persist the index.

    Returns:
        HybridIndex object.
    """
    # TODO: implement hybrid index construction
    pass


def load_index(
    collection_name: str = "ci_trials",
    persist_dir: str | None = None,
) -> HybridIndex:
    """
    Load a persisted hybrid index from disk.

    Args:
        collection_name: Name of the index collection.
        persist_dir: Directory where the index was persisted.

    Returns:
        HybridIndex object.
    """
    # TODO: implement index loading
    pass


def save_index(
    index: HybridIndex,
    collection_name: str = "ci_trials",
    persist_dir: str | None = None,
) -> None:
    """
    Persist a hybrid index to disk.

    Args:
        index: HybridIndex object to save.
        collection_name: Name for the index collection.
        persist_dir: Directory to persist the index.
    """
    # TODO: implement index persistence
    pass
