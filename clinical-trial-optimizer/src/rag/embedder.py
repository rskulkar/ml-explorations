"""Embedding module for CI RAG pipeline using PubMedBERT.

Provides:
    - load_pubmedbert: Load pre-trained PubMedBERT sentence transformer.
    - embed_chunks: Generate dense embeddings for criterion chunks.
"""
import numpy as np
from sentence_transformers import SentenceTransformer


def load_pubmedbert() -> SentenceTransformer:
    """
    Load the PubMedBERT-based sentence transformer for biomedical text.

    Returns:
        SentenceTransformer model (BioLink-Transformers or pubmedbert variant).
    """
    print("Loading PubMedBERT...")
    model = SentenceTransformer("microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract")
    return model


def embed_chunks(
    chunks: list,
    model: SentenceTransformer | None = None,
) -> np.ndarray:
    """
    Generate dense embeddings for a list of criterion chunks.

    Args:
        chunks: List of CriterionChunk objects.
        model: Optional pre-loaded SentenceTransformer. If None, loads pubmedbert.

    Returns:
        numpy array of shape (len(chunks), embedding_dim).
    """
    if model is None:
        model = load_pubmedbert()

    texts = [chunk.text for chunk in chunks]
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)
    return embeddings
