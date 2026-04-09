# Clinical Trial Optimizer

Multi-agent framework for I/E criteria analysis using real-world data (RWD) and competitive intelligence.

## Agents

| Agent | File | Input | Output |
|-------|------|-------|--------|
| SOC Agent | `src/agent_prompt_functions.py` | `ie_criteria`, `soc_df` | Gap table vs current standard of care |
| ET Agent | `src/agent_prompt_functions.py` | `ie_criteria`, `et_md_df`, period ranges | Gap table vs evolving treatment trends |
| CI Agent | `src/ci_agent_prompt.py` | `ie_criteria`, `competing_trials_md` | Gap table vs competing trials |

## Data Flow

```
Raw RWD input
    └─► data/raw/
         │
         ├─► filterSOCLOT / formatSOCLotDF (data_utils.py)
         │        └─► data/processed/   ──► SOC Agent prompt ──► SOC gap table
         │
         ├─► filterETLOT / formatETLotDF (data_utils.py)
         │        └─► data/processed/   ──► ET Agent prompt  ──► ET gap table
         │
         └─► fetch_competing_trials (ct_api.py)
                  └─► data/trials/      ──► CI Agent prompt  ──► CI gap table
                                                  │
                                         orchestrator.py
                                                  │
                                         Combined output (merged gap tables)
```

## Project Structure

```
clinical-trial-optimizer/
├── data/
│   ├── raw/          # Raw RWD input files
│   ├── processed/    # Filtered and formatted DataFrames
│   └── trials/       # ClinicalTrials.gov API results
├── notebooks/
│   ├── 01_soc_agent.ipynb
│   ├── 02_et_agent.ipynb
│   ├── 03_ci_agent.ipynb
│   └── 04_combined_output.ipynb
├── src/
│   ├── __init__.py
│   ├── agent_prompt_functions.py  # SOC + ET agent prompts
│   ├── ci_agent_prompt.py         # CI agent prompt
│   ├── ct_api.py                  # ClinicalTrials.gov v2 wrapper
│   ├── data_utils.py              # filterSOCLOT, filterETLOT, formatters
│   └── orchestrator.py            # Three-agent pipeline coordinator
└── tests/
    └── test_clinical_trial_optimizer.py
```

## Requirements

- `ANTHROPIC_API_KEY` must be set in the environment for agent cells.
- Internet access required for `ct_api.py` (ClinicalTrials.gov v2 API).
