"""Orchestrator for CI RAG pipeline end-to-end execution.

Provides:
    - run_ci_rag: Execute the full CI RAG pipeline (chunking → embedding → indexing → retrieval → reasoning → evaluation).
"""
import hashlib
import logging
from pathlib import Path

import pandas as pd

from chunker import chunk_competing_trials, parse_criteria_via_llm
from ci_reasoner import reason_about_criterion
from embedder import embed_chunks, load_pubmedbert
from index import build_index, load_index
from recommendation_evaluator import evaluate_recommendations, format_report
from retrieval_evaluator import evaluate_retrieval
from retriever import retrieve

logger = logging.getLogger(__name__)


def run_ci_rag(
    ie_criteria: str,
    competing_trials_md: str,
    persist_dir: str | None = None,
    api_key: str | None = None,
) -> pd.DataFrame:
    """
    Execute the full CI RAG pipeline end-to-end.

    Steps:
    1. Parse I/E criteria into source criterion chunks.
    2. Chunk competing trials from markdown table.
    3. Embed all chunks using PubMedBERT.
    4. Build hybrid index (FAISS + BM25).
    5. For each source criterion, retrieve relevant competing trial criteria.
    6. Reason about whether to KEEP/RELAX/TIGHTEN the criterion using Claude Sonnet.
    7. Evaluate recommendations for consistency and confidence.

    Args:
        ie_criteria: Raw I/E criteria text from the source trial.
        competing_trials_md: Markdown table of competing trials (from fetch_competing_trials).
        persist_dir: Optional directory to persist the index for reuse.
        api_key: Optional Anthropic API key.

    Returns:
        pandas DataFrame with columns: criterion_type, criterion_text, label, rationale, evidence_trials, suggested_wording, confidence, precision_at_k, mrr, coverage
    """
    logger.info("Starting CI RAG pipeline...")

    # Step 1: Chunk competing trials
    logger.info("Step 1: Chunking competing trials...")
    competitor_chunks = chunk_competing_trials(competing_trials_md, api_key)
    if not competitor_chunks:
        logger.warning("No competitor criteria extracted. Returning empty DataFrame.")
        return pd.DataFrame(columns=[
            "criterion_type", "criterion_text", "label", "rationale", "evidence_trials",
            "suggested_wording", "confidence", "precision_at_k", "mrr", "coverage"
        ])

    logger.info(f"Extracted {len(competitor_chunks)} competing trial criteria")

    # Step 2: Load PubMedBERT model
    logger.info("Step 2: Loading embedding model...")
    model = load_pubmedbert()

    # Step 3: Embed all chunks
    logger.info("Step 3: Embedding chunks...")
    embeddings = embed_chunks(competitor_chunks, model)

    # Step 4: Build or load hybrid index
    logger.info("Step 4: Building/loading hybrid index...")
    collection_name = "ci_rag_" + hashlib.md5(competing_trials_md[:100].encode()).hexdigest()

    hybrid_index = None
    if persist_dir:
        persist_path = Path(persist_dir) / collection_name
        if persist_path.exists():
            logger.info(f"Loading existing index from {persist_path}")
            hybrid_index = load_index(collection_name, persist_dir)

    if hybrid_index is None:
        hybrid_index = build_index(competitor_chunks, embeddings, collection_name, persist_dir)
        logger.info(f"Built new index with {len(competitor_chunks)} chunks")

    # Step 5: Parse source criteria (treatment-related only)
    logger.info("Step 5: Parsing source I/E criteria...")
    source_chunks = parse_criteria_via_llm(ie_criteria, "SOURCE", "Source Trial", {}, api_key)
    # Filter to treatment-related criteria
    source_chunks = [c for c in source_chunks if c.metadata.get("treatment_related", False)]
    logger.info(f"Extracted {len(source_chunks)} treatment-related source criteria")

    if not source_chunks:
        logger.warning("No treatment-related criteria in source. Returning empty DataFrame.")
        return pd.DataFrame(columns=[
            "criterion_type", "criterion_text", "label", "rationale", "evidence_trials",
            "suggested_wording", "confidence", "precision_at_k", "mrr", "coverage"
        ])

    # Step 6: Retrieve and reason for each source criterion
    logger.info("Step 6: Retrieving and reasoning about criteria...")
    recommendations = []
    retrieval_metrics = []

    competitor_nct_ids = list(set(c.source_nct_id for c in competitor_chunks))

    for i, source_chunk in enumerate(source_chunks, start=1):
        logger.info(f"  [{i}/{len(source_chunks)}] Retrieving for: {source_chunk.text[:60]}...")

        # Retrieve relevant competing trial criteria
        results = retrieve(source_chunk.text, hybrid_index, model, k=10)

        # Evaluate retrieval quality
        metrics = evaluate_retrieval(results, competitor_nct_ids, k=10)
        retrieval_metrics.append(metrics)

        # Reason about the criterion
        recommendation = reason_about_criterion(source_chunk.text, results, api_key)
        recommendations.append({
            "source_chunk": source_chunk,
            "recommendation": recommendation,
            "metrics": metrics,
        })

    # Step 7: Evaluate all recommendations
    logger.info("Step 7: Evaluating recommendations...")
    rec_list = [r["recommendation"] for r in recommendations]
    eval_report = evaluate_recommendations(rec_list)

    # Build output DataFrame
    rows = []
    for item in recommendations:
        source_chunk = item["source_chunk"]
        rec = item["recommendation"]
        metrics = item["metrics"]

        rows.append({
            "criterion_type": source_chunk.criterion_type,
            "criterion_text": source_chunk.text,
            "label": rec.label,
            "rationale": rec.rationale,
            "evidence_trials": ", ".join(rec.evidence_trials),
            "suggested_wording": rec.suggested_wording,
            "confidence": rec.confidence,
            "precision_at_k": metrics.precision_at_k,
            "mrr": metrics.mrr,
            "coverage": metrics.coverage,
        })

    df = pd.DataFrame(rows)

    # Print summary
    logger.info("\n" + format_report(eval_report))

    logger.info("CI RAG pipeline complete.")
    return df
