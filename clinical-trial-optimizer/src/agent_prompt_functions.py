"""SOC and ET agent prompt functions for clinical-trial-optimizer.

Provides:
    - soc_agent_prompt: builds the prompt for the Standard-of-Care (SOC) agent,
      which compares I/E criteria against a current standard-of-care DataFrame (soc_df)
      and returns a gap table.
    - et_agent_prompt: builds the prompt for the Evolving-Treatment (ET) agent,
      which compares I/E criteria against an evolving-treatment DataFrame (et_md_df)
      over specified period ranges and returns a gap table.
"""
