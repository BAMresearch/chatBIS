This reference lists the on-disk data layout used by chatBIS.

# Data layout

## Raw data

Default path: `<cwd>\data\raw`

The scraper stores one text file per page. Each file contains:

- `Title: ...`
- `URL: ...`
- A `---` separator
- Extracted content as markdown-like text

## Processed data

Default path: `<cwd>\data\processed`

Files produced by the processor:

- `chunks.json` - chunk records with content and embeddings
- `chunks.csv` - chunk metadata without embeddings

Each JSON chunk includes a `chunk_id`, `title`, `url`, `content`, and `embedding` vector.

## Conversation memory

When using the CLI or web UI with memory, chatBIS stores a SQLite database at:

- `<data_dir>\conversation_memory.db`
