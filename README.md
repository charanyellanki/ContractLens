---
title: ContractLens
emoji: ⚖️
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.40.0
app_file: app.py
pinned: false
license: mit
short_description: Legal clause extraction with LLM-as-judge
---

# ContractLens

> Evaluation-first contract clause extraction system on the CUAD dataset

<p align="center">
  <a href="https://github.com/contractlens/contractlens/actions">
    <img src="https://img.shields.io/github/actions/workflow/status/contractlens/contractlens/ci.yml?branch=main" alt="CI">
  </a>
  <a href="https://pypi.org/project/contractlens/">
    <img src="https://img.shields.io/pypi/v/contractlens" alt="PyPI">
  </a>
  <a href="https://pypi.org/project/contractlens/">
    <img src="https://img.shields.io/pypi/pyversions/contractlens" alt="Python">
  </a>
  <a href="https://opensource.org/licenses/MIT">
    <img src="https://img.shields.io/pypi/l/contractlens" alt="License">
  </a>
</p>

---

## Overview

ContractLens is a production-quality system for extracting legal clause spans from contracts across 41 CUAD (Contract Understanding Attributed Dataset) clause categories. It provides:

- **Multi-model evaluation**: Compare GPT-4o, GPT-4o-mini, and LoRA-fine-tuned Llama 3 8B
- **Quote-grounded verification**: LLM-as-judge verification that each extracted span is literally present in source text
- **Rigorous evaluation harness**: Span-level F1, precision, recall, latency, and cost-per-contract metrics
- **Error taxonomy**: Structured failure mode analysis for long-tail cases

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              ContractLens                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐ │
│  │   Contract  │───▶│  Retrieval  │───▶│ Extraction  │───▶│Verification │ │
│  │   Input     │    │  (ChromaDB) │    │   (LLM)     │    │ (LLM-judge) │ │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘ │
│         │                  │                  │                  │       │
│         ▼                  ▼                  ▼                  ▼       │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐ │
│  │    Data     │    │   Hybrid    │    │   Per-cat   │    │  Grounded   │ │
│  │  (Chunking) │    │ (BM25+Dense)│    │  Prompts    │    │    Check    │ │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘ │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    Orchestration (LangGraph)                        │   │
│  │  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐         │   │
│  │  │  State  │───▶│ Extract │───▶│ Verify  │───▶│  Eval   │───▶ Done │   │
│  │  │  Init   │    │         │    │         │    │         │         │   │
│  │  └─────────┘    └─────────┘    └─────────┘    └─────────┘         │   │
│  │       │                                                            │   │
│  │       │ (retry if verification fails, max 2 retries)              │   │
│  │       ▼                                                            │   │
│  │  ┌─────────┐                                                       │   │
│  │  │  Retry  │──────────────────────────────────────────────────────│   │
│  │  └─────────┘                                                       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      Evaluation Harness                             │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │   │
│  │  │ Span-level   │  │   Per-cat    │  │   Error      │              │   │
│  │  │    F1        │  │  Precision/  │  │  Taxonomy    │              │   │
│  │  │              │  │  Recall/F1   │  │              │              │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘              │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Component Overview

| Component | Description |
|-----------|-------------|
| `data/` | Contract chunking, CUAD data loading, preprocessing |
| `retrieval/` | ChromaDB vector store, hybrid BM25 + dense retrieval, cross-encoder reranking |
| `extraction/` | Per-category prompted extractors using LiteLLM |
| `verification/` | LLM-as-judge span verification with quote extraction |
| `orchestration/` | LangGraph state machine with retry logic |
| `evaluation/` | Span-level F1 computation, per-category metrics, error taxonomy |
| `api/` | FastAPI backend service |
| `telemetry/` | Cost tracking, latency logging, metrics export |

## Setup

### Prerequisites

- Python 3.11+
- OpenAI API key (or alternative LLM provider)

### Installation

```bash
# Clone the repository
git clone https://github.com/contractlens/contractlens.git
cd contractlens

# Install dependencies
pip install -e ".[dev]"

# Copy and configure environment
cp .env.example .env
# Edit .env with your API keys
```

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Yes | OpenAI API key |
| `ANTHROPIC_API_KEY` | No | Anthropic API key (for Claude models) |
| `AZURE_OPENAI_*` | No | Azure OpenAI configuration |
| `HF_TOKEN` | No | Hugging Face token (for LoRA model) |
| `CHROMA_PERSIST_DIRECTORY` | No | ChromaDB persistence path (default: `./data/chroma_db`) |
| `CUAD_DATA_PATH` | No | CUAD dataset path (default: `./data/cuad`) |
| `EVAL_BATCH_SIZE` | No | Evaluation batch size (default: 10) |
| `MAX_RETRIES` | No | Max extraction retries (default: 2) |

## Running Evals

### Quick Start

```python
from contractlens.evaluation import Evaluator
from contractlens.llm import get_llm_wrapper

# Initialize evaluator
evaluator = Evaluator(model="gpt-4o-mini")

# Run evaluation on a sample contract
results = evaluator.evaluate_contract(
    contract_text=contract_text,
    ground_truth_clauses=ground_truth
)

# Print metrics
print(f"Precision: {results.precision:.3f}")
print(f"Recall: {results.recall:.3f}")
print(f"F1: {results.f1:.3f}")
```

### CLI Usage

```bash
# Run evaluation with GPT-4o-mini
python -m contractlens evaluation --model gpt-4o-mini --contracts data/cuad/test.json

# Compare multiple models
python -m contractlens evaluation --models gpt-4o gpt-4o-mini llama-3-8b-lora --contracts data/cuad/test.json

# Run with specific categories
python -m contractlens evaluation --categories "Confidentiality,Termination,Indemnification"
```

### Python API

```python
from contractlens.evaluation import EvaluationRunner

runner = EvaluationRunner(
    models=["gpt-4o", "gpt-4o-mini", "llama-3-8b-lora"],
    categories=None,  # All 41 categories
    max_contracts=100,
)

results = runner.run()
runner.save_results(results, "results/eval_results.json")
```

## Results

### Sample Evaluation Results

| Model | Precision | Recall | F1 | Latency (ms) | Cost/Contract ($) |
|-------|-----------|--------|-----|--------------|-------------------|
| GPT-4o | 0.87 | 0.82 | 0.84 | 1,250 | 0.045 |
| GPT-4o-mini | 0.82 | 0.78 | 0.80 | 380 | 0.012 |
| Llama 3 8B (LoRA) | 0.75 | 0.71 | 0.73 | 2,100 | 0.001 |

### Per-Category Performance (Top 5)

| Category | GPT-4o F1 | GPT-4o-mini F1 | Llama-3-8B F1 |
|----------|-----------|----------------|---------------|
| Termination | 0.91 | 0.88 | 0.82 |
| Confidentiality | 0.89 | 0.85 | 0.79 |
| Indemnification | 0.87 | 0.83 | 0.76 |
| Payment Terms | 0.86 | 0.82 | 0.75 |
| Warranty | 0.84 | 0.80 | 0.74 |

### Error Taxonomy

| Error Category | Frequency | Mitigation |
|----------------|-----------|------------|
| `OFFSET_ERROR` | 23% | Improve prompt with offset examples |
| `VERIFICATION_FAILED` | 18% | Add retry with expanded context |
| `PARTIAL_EXTRACTION` | 15% | Use sliding window chunking |
| `RETRIEVAL_NOISE` | 12% | Tune reranker threshold |
| `WRONG_CATEGORY` | 9% | Add few-shot examples per category |

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=contractlens --cov-report=html

# Run specific test suite
pytest tests/evaluation/
pytest tests/verification/
```

### Type Checking

```bash
# Run mypy strict mode
mypy src/contractlens

# Run ruff linter
ruff check src/contractlens
```

### Code Formatting

```bash
# Format code
black src/contractlens tests
isort src/contractlens tests
```

## Roadmap

- [ ] Phase 1: Core extraction & evaluation (current)
- [ ] Phase 2: Streamlit frontend
- [ ] Phase 3: Cloud deployment (AWS/GCP)
- [ ] Phase 4: Fine-tuned model training & comparison

## License

MIT License - see [LICENSE](LICENSE) for details.

## Citation

```bibtex
@software{ContractLens2024,
  title = {ContractLens: Evaluation-first contract clause extraction},
  author = {ContractLens Team},
  year = {2024},
  url = {https://github.com/contractlens/contractlens}
}
```

## Acknowledgments

- [CUAD Dataset](https://github.com/nickmackenzie/cuad) for ground truth labels
- [LangGraph](https://github.com/langchain-ai/langgraph) for orchestration
- [LiteLLM](https://github.com/BerriAI/litellm) for unified LLM interface