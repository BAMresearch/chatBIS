This explanation describes why chatBIS stores certain data locally and how to handle credentials safely.

# Security and privacy

chatBIS is designed for local-first use, but it still stores data on disk and can connect to external services.

## What is stored locally

- Scraped documentation under `data/raw`
- Processed chunks and embeddings under `data/processed`
- Conversation history and session IDs in `conversation_memory.db`
- Logs printed to stdout

## Credential handling

- Credentials can be loaded from environment variables (`OPENBIS_URL`, `OPENBIS_USERNAME`, `OPENBIS_PASSWORD`).
- If you provide credentials via chat messages, they become part of the conversation history stored in SQLite.

## Network access

- Scraping performs HTTP requests to the documentation site.
- pybis tools connect to the configured openBIS server.
- Ollama is expected to run locally for embeddings and chat.
