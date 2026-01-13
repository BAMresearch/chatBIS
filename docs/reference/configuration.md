This reference lists configuration defaults and environment variables.

# Configuration

## Default paths

| Setting | Default | Source |
| --- | --- | --- |
| Data root | `<cwd>\data` | `chatBIS.__main__` |
| Raw data | `<cwd>\data\raw` | `chatBIS.__main__` |
| Processed data | `<cwd>\data\processed` | `chatBIS.__main__` and `chatBIS.web.cli` |
| Memory DB | `<data_dir>\conversation_memory.db` | `chatBIS.query.cli` and `chatBIS.web.app` |

## Default URLs and limits

| Setting | Default | Source |
| --- | --- | --- |
| Scrape URL (auto mode) | `https://openbis.readthedocs.io/en/latest/` | `chatBIS.__main__` |
| Max pages (auto mode) | 100 | `chatBIS.__main__` |
| Request delay | 0.5 seconds | `chatBIS.scraper.cli` |

## Model defaults

| Setting | Default | Source |
| --- | --- | --- |
| Chat model (CLI) | `qwen3` | `chatBIS.query.cli` |
| Chat model (web CLI) | `qwen3` | `chatBIS.web.cli` |
| Chat model (web app direct) | `gpt-oss:20b` | `chatBIS.web.app` |
| Chat model (RAGQueryEngine default) | `gpt-oss:20b` | `chatBIS.query.query` |
| Embedding model | `nomic-embed-text` | `chatBIS.processor.processor` and `chatBIS.query.query` |

## Environment variables

| Variable | Purpose | Used by |
| --- | --- | --- |
| `OPENBIS_URL` | openBIS server URL for auto-connect | `chatBIS.tools.pybis_tools` |
| `OPENBIS_USERNAME` | openBIS username for auto-connect and default space | `chatBIS.tools.pybis_tools` |
| `OPENBIS_PASSWORD` | openBIS password for auto-connect | `chatBIS.tools.pybis_tools` |
| `SECRET_KEY` | Flask session key | `chatBIS.web.app` |

`python-dotenv` is loaded in the query and processor CLIs and in the pybis tools module, so `.env` files are supported.
