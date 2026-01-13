This page introduces chatBIS and points you to the tutorials, how-to guides, reference, and explanations.

# chatBIS

chatBIS is a Retrieval Augmented Generation (RAG) chatbot for openBIS documentation with persistent memory, a CLI, and a web UI.

## Key features

- RAG over scraped openBIS documentation
- Conversation memory with session IDs stored in SQLite
- Multi-agent routing between documentation answers and pybis actions
- CLI and Flask-based web interface
- Local-first workflow using Ollama models

## Quickstart

1. Install prerequisites: Python 3.8+ and Ollama.
2. Pull the models used by default:

```bash
ollama pull nomic-embed-text
ollama pull qwen3
```

3. From the repository root, install and run:

```bash
pip install -e .
python -m chatBIS
```

The command checks for `data/processed/chunks.json`. If it is missing, it automatically scrapes and processes the docs before starting the CLI.

## Where to go next

- Tutorials: `tutorials/index.md`
- How-to guides: `howtos/index.md`
- Reference: `reference/index.md`
- Explanations: `explanations/index.md`
