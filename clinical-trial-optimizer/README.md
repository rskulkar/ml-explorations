# Clinical Trial Optimizer

Multi-agent framework for I/E criteria analysis using real-world data (RWD) and competitive intelligence.

## Agents

| Agent | File | Input | Output |
|-------|------|-------|--------|
| SOC Agent | `src/agent_prompt_functions.py` | `ie_criteria`, `soc_df` | Gap table vs current standard of care |
| ET Agent | `src/agent_prompt_functions.py` | `ie_criteria`, `et_md_df`, period ranges | Gap table vs evolving treatment trends |
| CI Agent | `src/ci_agent_prompt.py` | `ie_criteria`, `competing_trials_md` | Gap table vs competing trials |

## Advanced Modular RAG Pipeline (CI Agent)

Production-ready hybrid RAG pipeline for competitive intelligence analysis of clinical trial I/E criteria.

### Architecture
```
ClinicalTrials.gov API → Criterion Chunker → PubMedBERT Embedder → Hybrid Index
│
┌────────────────────┤
│                    │
FAISS (dense)        BM25 (sparse)
│                    │
└──── RRF Fusion ────┘
│
CI Reasoner (Sonnet)
│
KEEP / RELAX / TIGHTEN
``` 

### Modules

| Module | File | Purpose |
|--------|------|---------|
| Chunker | `src/rag/chunker.py` | LLM-based criterion extraction via Haiku — one chunk per criterion |
| Embedder | `src/rag/embedder.py` | PubMedBERT dense embeddings (768-dim) |
| Index | `src/rag/index.py` | FAISS IndexFlatIP + BM25Okapi, persisted to disk |
| Retriever | `src/rag/retriever.py` | Hybrid retrieval with Reciprocal Rank Fusion (k=60) |
| Retrieval Evaluator | `src/rag/retrieval_evaluator.py` | Precision@K, MRR, coverage metrics |
| CI Reasoner | `src/rag/ci_reasoner.py` | Claude Sonnet clinical reasoning — KEEP/RELAX/TIGHTEN |
| Recommendation Evaluator | `src/rag/recommendation_evaluator.py` | Aggregate stats and consistency scoring |
| Pipeline | `src/rag/ci_rag_pipeline.py` | End-to-end orchestrator |

### Design Decisions

- **Domain-specialised embeddings:** PubMedBERT trained on PubMed abstracts outperforms general-purpose models on clinical text
- **Hybrid retrieval:** Dense search captures semantic similarity; BM25 captures drug name and clinical term exact matches; RRF fusion is robust and parameter-free
- **Criterion-level chunking:** Each I/E criterion is a separate chunk — preserves clinical meaning and enables criterion-to-criterion comparison
- **All criteria evaluated:** Unlike SOC/ET agents which filter to treatment-related criteria only, CI evaluates all criteria types
- **Persisted index:** FAISS index and BM25 corpus saved to `data/indexes/` between sessions

### Output Labels

- **KEEP** — criterion aligned with or more rigorous than competing trials
- **RELAX** — criterion more restrictive than competing trials; consider broadening
- **TIGHTEN** — criterion less restrictive than competing trials; consider narrowing

## Data Flow
```
Raw RWD input
└─► data/raw/
│
├─► filterSOCLOT / formatSOCLotDF (data_utils.py)
│        └─► SOC Agent prompt ──► SOC gap table
│
├─► filterETLOT / formatETLotDF (data_utils.py)
│        └─► ET Agent prompt  ──► ET gap table
│
└─► fetch_competing_trials (ct_api.py)
└─► CI Agent prompt  ──► CI RAG pipeline ──► CI gap table
│
orchestrator.py
│
Combined output
```

## Usage

```python
from src.agent_prompt_functions import get_soc_agent_prompt, get_et_agent_prompt
from src.agent_prompt_functions import get_ci_agent_prompt
from src.ct_api import fetch_competing_trials
from src.data_utils import filterSOCLOT, formatSOCLotDF, filterETLOT, formatETLotDF
from src.rag.ci_rag_pipeline import run_ci_rag

# SOC and ET agents
soc_df = formatSOCLotDF(filterSOCLOT(df))
et_df = formatETLotDF(filterETLOT(df1, df2))
soc_prompt = get_soc_agent_prompt(ie_criteria, soc_df)
et_prompt = get_et_agent_prompt(ie_criteria, et_df)

# CI RAG pipeline
competing_trials_md = fetch_competing_trials(ie_criteria)
results_df = run_ci_rag(ie_criteria, competing_trials_md, persist_dir="data/indexes/")
```

## Project Structure
```
clinical-trial-optimizer/
├── data/
│   ├── raw/          # Raw RWD input files
│   ├── processed/    # Filtered and formatted DataFrames
│   ├── trials/       # ClinicalTrials.gov API results
│   └── indexes/      # Persisted FAISS + BM25 indexes
├── notebooks/
│   ├── 01_soc_agent.ipynb
│   ├── 02_et_agent.ipynb
│   ├── 03_ci_agent.ipynb
│   ├── 04_combined_output.ipynb
│   └── 05_ci_rag_pipeline.ipynb
├── src/
│   ├── agent_prompt_functions.py  # SOC + ET + CI agent prompts
│   ├── ct_api.py                  # ClinicalTrials.gov v2 wrapper
│   ├── data_utils.py              # filterSOCLOT, filterETLOT, formatters
│   ├── orchestrator.py            # Three-agent pipeline coordinator
│   └── rag/
│       ├── chunker.py
│       ├── embedder.py
│       ├── index.py
│       ├── retriever.py
│       ├── retrieval_evaluator.py
│       ├── ci_reasoner.py
│       ├── recommendation_evaluator.py
│       └── ci_rag_pipeline.py
└── tests/
└── test_clinical_trial_optimizer.py
```

## Migration Checklist

- [x] Copy `agent_prompt_functions.py` → `src/agent_prompt_functions.py`
- [x] Migrate `filterSOCLOT`, `formatSOCLotDF`, `filterETLOT`, `formatETLotDF` → `src/data_utils.py`
- [x] Migrate `fetch_competing_trials` → `src/ct_api.py`
- [x] Replace LiteLLM proxy client with direct anthropic SDK
- [x] Implement Advanced Modular RAG pipeline in `src/rag/`
- [ ] Add real unit tests to `test_clinical_trial_optimizer.py`
- [ ] Wire `orchestrator.py` to combine SOC + ET + CI outputs

## Environment

- Client: anthropic SDK (not LiteLLM proxy)
- Set `ANTHROPIC_API_KEY` in `~/.zprofile`
- Internet access required for `ct_api.py`