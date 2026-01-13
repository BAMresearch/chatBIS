This tutorial will get you from a fresh clone to your first chatBIS run.

# Getting started

## Prerequisites

- Python 3.8 or newer
- Ollama running locally
- Models pulled in Ollama:

```bash
ollama pull nomic-embed-text
ollama pull qwen3
```

## Install the package

From the repository root:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

## First run (auto mode)

Run the main entrypoint with no arguments:

```bash
python -m chatBIS
```

If `data/processed/chunks.json` is missing, chatBIS will scrape and process the documentation automatically, then start the CLI.
