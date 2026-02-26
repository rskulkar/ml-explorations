# CLAUDE.md

## Project Overview
Monorepo for three parallel ML projects exploring Claude Code, Plan Mode, and MCP integrations.

## Repository Structure
```
ml-explorations/
├── .claude/mcp_config.json
├── shared/utils/
├── mmm-marketing/
├── chronometer-mimic/
├── content-moderation/
└── pyproject.toml
```

## Projects

### 1. mmm-marketing
Media Mix Modeling comparing PyMC-Marketing vs Stan (v2).
- Dataset: Robyn open-source dataset (Meta)
- Goal: Channel allocation sensitivity comparison across approaches
- Stack: Python, PyMC-Marketing, ArviZ, pandas

### 2. chronometer-mimic
Transformer-based 30-day readmission model on MIMIC-IV-Demo.
- Dataset: MIMIC-IV-Demo (100 patients, no credentials required)
- Preprocessing: MEDS-Transforms library
- Goal: Validate pipeline architecture against Chronometer paper
- Stack: Python, PyTorch (MPS), MEDS-Transforms, HuggingFace

### 3. content-moderation
Content moderation audit for Indian English tweets.
- Training data: Academic corpus (Kaggle/HuggingFace labeled datasets)
- Live test data: 500 tweets via Twitter Free tier API
- Architecture: Blocklist (Lumen) + ML classifier + RAG = Combined Architecture
- Stack: Python, scikit-learn, sentence-transformers, ChromaDB, tweepy

## Environment
- Hardware: Apple M1 8GB RAM, MPS available for GPU acceleration
- Python: managed via uv
- Node: v20 via nvm

## Credentials
All secrets loaded from environment variables — never hardcoded.
- GITHUB_PAT
- TWITTER_BEARER_TOKEN
- TWITTER_API_KEY
- TWITTER_API_SECRET

## Build Commands
```bash
uv sync                    # install dependencies
uv run pytest             # run all tests
uv run pytest mmm-marketing/tests/    # project-specific tests
```

## Development Principles
- Write unit tests before implementation
- Keep batch sizes conservative for M1 8GB (max 16 for transformers)
- All notebooks in notebooks/ subdirectory per project
- No credentials in code or notebooks
- Human approval required before: API calls, model training runs, external data downloads

## MCP Servers
Configured in .claude/mcp_config.json:
- filesystem: scoped to repo root
- github: for repo management via GITHUB_PAT
- fetch: for Lumen Database and Twitter API calls
