"""Hybrid index (FAISS + BM25) module for CI RAG pipeline.

Provides:
    - HybridIndex: dataclass combining dense (FAISS) and BM25 indexes.
    - build_index: Construct and persist a hybrid index.
    - load_index: Load a persisted hybrid index.
    - save_index: Persist a hybrid index to disk.
"""
import json
import logging
import pickle
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


@dataclass
class HybridIndex:
    """Hybrid index combining FAISS (dense) and BM25 (sparse) retrieval."""
    faiss_index: Optional[object] = None  # faiss.IndexFlatIP
    bm25: Optional[BM25Okapi] = None
    chunks: list = field(default_factory=list)
    embeddings: Optional[np.ndarray] = None


def build_index(
    chunks: list,
    embeddings: np.ndarray,
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
    # Normalise embeddings for cosine similarity (IndexFlatIP uses inner product on normalised vectors)
    faiss.normalize_L2(embeddings)

    # Build FAISS index (IndexFlatIP = inner product, works on normalised vectors)
    embedding_dim = embeddings.shape[1]
    faiss_index = faiss.IndexFlatIP(embedding_dim)
    faiss_index.add(embeddings.astype(np.float32))

    # Build BM25 index from tokenised chunk texts
    tokenized_texts = [chunk.text.lower().split() for chunk in chunks]
    bm25 = BM25Okapi(tokenized_texts)

    # Create HybridIndex
    hybrid_index = HybridIndex(
        faiss_index=faiss_index,
        bm25=bm25,
        chunks=chunks,
        embeddings=embeddings,
    )

    # Persist if directory provided
    if persist_dir is not None:
        save_index(hybrid_index, collection_name, persist_dir)

    return hybrid_index


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
    if persist_dir is None:
        return

    persist_path = Path(persist_dir) / collection_name
    persist_path.mkdir(parents=True, exist_ok=True)

    # Save FAISS index
    faiss.write_index(index.faiss_index, str(persist_path / "faiss.index"))

    # Save BM25 corpus and metadata (pickle)
    with open(persist_path / "bm25_corpus.pkl", "wb") as f:
        pickle.dump({"bm25": index.bm25}, f)

    # Save chunks metadata as JSON
    chunks_metadata = [asdict(chunk) for chunk in index.chunks]
    # Remove embedding field (large array) from metadata
    for cm in chunks_metadata:
        cm.pop("embedding", None)
    with open(persist_path / "chunks_metadata.json", "w") as f:
        json.dump(chunks_metadata, f, indent=2)

    logger.info(f"Index saved to {persist_path}")


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
    if persist_dir is None:
        raise ValueError("persist_dir must be provided to load_index")

    persist_path = Path(persist_dir) / collection_name
    if not persist_path.exists():
        raise FileNotFoundError(f"Index collection '{collection_name}' not found at {persist_path}")

    # Load FAISS index
    faiss_index = faiss.read_index(str(persist_path / "faiss.index"))

    # Load BM25 and corpus
    with open(persist_path / "bm25_corpus.pkl", "rb") as f:
        bm25_data = pickle.load(f)
    bm25 = bm25_data["bm25"]

    # Load chunks metadata
    with open(persist_path / "chunks_metadata.json", "r") as f:
        chunks_metadata = json.load(f)

    # Reconstruct CriterionChunk objects
    from chunker import CriterionChunk

    chunks = []
    for cm in chunks_metadata:
        chunk = CriterionChunk(
            text=cm["text"],
            criterion_type=cm["criterion_type"],
            source_nct_id=cm.get("source_nct_id"),
            source_trial_name=cm.get("source_trial_name"),
            trial_phase=cm.get("trial_phase"),
            trial_status=cm.get("trial_status"),
            trial_sponsor=cm.get("trial_sponsor"),
            metadata=cm.get("metadata", {}),
        )
        chunks.append(chunk)

    logger.info(f"Index loaded from {persist_path}")
    return HybridIndex(faiss_index=faiss_index, bm25=bm25, chunks=chunks)
