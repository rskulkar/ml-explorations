"""Data utility functions for clinical-trial-optimizer.

Provides:
    - filterSOCLOT: filters a standard-of-care lines-of-therapy DataFrame by
      indication and line-of-therapy criteria.
    - formatSOCLotDF: formats a filtered SOC DataFrame into markdown suitable
      for agent prompts.
    - filterETLOT: filters an evolving-treatment lines-of-therapy DataFrame by
      indication, treatment period, and date range.
    - formatETLotDF: formats a filtered ET DataFrame into markdown suitable
      for agent prompts.
"""
