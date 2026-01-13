This explanation describes why the chatBIS pipeline is split into scraper, processor, retrieval, and response stages.

# Architecture

chatBIS separates ingestion from chat-time retrieval so you can scrape and process documentation once, then query it quickly.

```mermaid
flowchart TD
  A[ReadTheDocs scraper] --> B[Raw text files]
  B --> C[Processor and chunker]
  C --> D[Processed chunks + embeddings]
  D --> E[RAG retrieval]
  E --> F[LLM (Ollama chat model)]
  F --> G[Response formatter]
  G --> H[SQLite memory store]
  H --> E
```

Key components:

- The scraper saves each page as a text file with title and URL metadata.
- The processor chunks content and generates embeddings.
- The conversation engine retrieves relevant chunks and routes requests to either RAG or pybis tools.
- Memory is persisted with LangGraph checkpoints in SQLite.
