"""Multi-agent orchestrator for clinical-trial-optimizer.

Coordinates the three-agent pipeline:
    1. SOC Agent  — gaps vs. current standard of care
    2. ET Agent   — gaps vs. evolving treatment trends
    3. CI Agent   — gaps vs. competing trials on ClinicalTrials.gov

Produces a combined output that merges the three gap tables for downstream
review and reporting.
"""
