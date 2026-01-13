This how-to shows quick fixes for common chatBIS issues.

# Troubleshoot

## Ollama is not running

- Start Ollama and retry.
- Verify models are present:

```bash
ollama pull nomic-embed-text
ollama pull qwen3
```

## No processed data found

Run the full pipeline or build it step-by-step:

```bash
python -m chatBIS
```

or

```bash
python -m chatBIS scrape --url https://openbis.readthedocs.io/en/latest/ --output .\data\raw
python -m chatBIS process --input .\data\raw --output .\data\processed
```

## SQLite permission issues

- Make sure the `--memory-db` path is writable.
- Use a custom path if the default data directory is read-only.

## Slow responses or retrieval tuning

The CLI does not expose retrieval size. Internally:

- The multi-agent RAG path retrieves `top_k=3` chunks in `chatBIS.query.conversation_engine`.
- The standalone `RAGQueryEngine.query()` default is `top_k=5` in `chatBIS.query.query`.

To change these values, update the code and rebuild your processed data if needed.

## pybis connection failures

- Confirm `OPENBIS_URL`, `OPENBIS_USERNAME`, and `OPENBIS_PASSWORD` are set.
- Try connecting explicitly in chat.
- Verify the server URL is reachable and credentials are correct.
