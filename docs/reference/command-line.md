This reference lists the chatBIS command-line interfaces and options.

# Command line

## Main entrypoint

Run the package entrypoint with:

```bash
python -m chatBIS --help
```

Top-level options:

| Argument | Type | Default | Description |
| --- | --- | --- | --- |
| `--web` | flag | false | Run the web interface instead of the CLI |

Subcommands:

- `scrape` - Scrape a ReadTheDocs site
- `process` - Process scraped content for RAG
- `query` - Chat with memory using processed data
- `web` - Run the web interface
- `auto` - Internal auto mode

## `scrape`

```bash
python -m chatBIS scrape --help
```

| Argument | Type | Default | Description |
| --- | --- | --- | --- |
| `--url` | string | required | The base URL of the ReadTheDocs site |
| `--output` | string | required | Directory to save scraped content |
| `--version` | string | none | Specific version to scrape, for example `en/latest` |
| `--delay` | float | 0.5 | Delay between requests in seconds |
| `--max-pages` | int | none | Maximum number of pages to scrape |
| `--verbose` | flag | false | Enable verbose logging |

## `process`

```bash
python -m chatBIS process --help
```

| Argument | Type | Default | Description |
| --- | --- | --- | --- |
| `--input` | string | required | Directory containing scraped content |
| `--output` | string | required | Directory to save processed content |
| `--api-key` | string | none | Not used for Ollama, kept for compatibility |
| `--min-chunk-size` | int | 100 | Minimum chunk size in characters |
| `--max-chunk-size` | int | 1000 | Maximum chunk size in characters |
| `--chunk-overlap` | int | 50 | Overlap between chunks in characters |
| `--verbose` | flag | false | Enable verbose logging |

## `query`

```bash
python -m chatBIS query --help
```

| Argument | Type | Default | Description |
| --- | --- | --- | --- |
| `--data` | string | required | Directory containing processed content |
| `--model` | string | qwen3 | Ollama model used for chat |
| `--memory-db` | string | `data/conversation_memory.db` | Path to SQLite database for conversation memory |
| `--session-id` | string | none | Session ID to continue a previous conversation |
| `--verbose` | flag | false | Enable verbose logging |

When `--memory-db` is omitted, the CLI constructs the path as `<data_dir>\conversation_memory.db`.

## `web`

```bash
python -m chatBIS web --help
```

| Argument | Type | Default | Description |
| --- | --- | --- | --- |
| `--data` | string | `.\data\processed` | Directory containing processed content |
| `--host` | string | 0.0.0.0 | Host to run the web interface on |
| `--port` | int | 5000 | Port to run the web interface on |
| `--model` | string | qwen3 | Ollama model used for chat |
| `--debug` | flag | false | Enable debug mode |
