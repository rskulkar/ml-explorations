# ml-explorations

Monorepo for three parallel ML projects exploring Claude Code, Plan Mode, and MCP integrations.

## Projects

### mmm-marketing
Media Mix Modeling comparing PyMC-Marketing vs Stan (v2).
- Dataset: Robyn open-source dataset (Meta)
- Goal: Channel allocation sensitivity comparison across approaches
- Stack: PyMC-Marketing, ArviZ, pandas

### chronometer-mimic
Transformer-based 30-day readmission model on MIMIC-IV-Demo.
- Dataset: MIMIC-IV-Demo (100 patients, no credentials required)
- Preprocessing: MEDS-Transforms library
- Goal: Validate pipeline architecture against Chronometer paper
- Stack: PyTorch (MPS), MEDS-Transforms, HuggingFace Transformers

### content-moderation
Content moderation audit for Indian English tweets.
- Training data: Academic corpus (Kaggle/HuggingFace labeled datasets)
- Live test data: 500 tweets via Twitter Free tier API
- Architecture: Blocklist (Lumen) + ML classifier + RAG = Combined Architecture
- Stack: scikit-learn, sentence-transformers, ChromaDB, tweepy

## Quick Start

```bash
# Install dependencies
uv sync

# Run all tests
uv run pytest

# Run tests for a single project
uv run pytest mmm-marketing/tests/
uv run pytest chronometer-mimic/tests/
uv run pytest content-moderation/tests/
```

## Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager
- Apple M1 (MPS available for GPU acceleration; batch sizes capped at 16 for transformers)

## Environment Variables

Copy `.env.example` (when created) and populate:

```
GITHUB_PAT=...
TWITTER_BEARER_TOKEN=...
TWITTER_API_KEY=...
TWITTER_API_SECRET=...
```

Never commit secrets to the repo.
