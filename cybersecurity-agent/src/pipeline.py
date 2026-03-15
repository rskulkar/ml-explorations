"""Orchestration pipeline for the cybersecurity-agent RAG gap analysis."""
from __future__ import annotations

import pathlib

import pandas as pd

import agent
import vectorstore as vs


def run_analysis(
    controls_dir: pathlib.Path,
    collection,
    k: int = 5,
) -> pd.DataFrame:
    """Run compliance gap analysis for all controls in a directory.

    For each .txt file in controls_dir:
    1. Reads the control text.
    2. Retrieves the top-k relevant standard chunks from the vector store.
    3. Calls the Claude agent to produce a gap analysis.

    Args:
        controls_dir: Directory containing .txt control files.
        collection: ChromaDB collection holding the embedded standard.
        k: Number of standard chunks to retrieve per control.

    Returns:
        DataFrame with columns:
            control_file, compliance_level, gaps, recommendations
    """
    controls_dir = pathlib.Path(controls_dir)
    control_files = sorted(controls_dir.glob("*.txt"))

    rows = []
    for control_path in control_files:
        control_text = control_path.read_text(encoding="utf-8")
        chunks = vs.query_vectorstore(collection, control_text, k=k)
        result = agent.analyse_control(control_text, chunks)
        rows.append(
            {
                "control_file": control_path.name,
                "compliance_level": result.get("compliance_level", ""),
                "gaps": result.get("gaps", []),
                "recommendations": result.get("recommendations", []),
            }
        )

    return pd.DataFrame(
        rows,
        columns=["control_file", "compliance_level", "gaps", "recommendations"],
    )
