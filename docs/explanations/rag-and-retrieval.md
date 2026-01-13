This explanation describes why chatBIS uses chunking and similarity search for retrieval.

# RAG and retrieval

chatBIS uses chunked documentation and embedding similarity to select relevant context for each query.

## Chunking

The processor splits content by headings and paragraph boundaries, then applies these defaults:

- Minimum chunk size: 100 characters
- Maximum chunk size: 1000 characters
- Chunk overlap: 50 characters

These values are configurable in the `process` CLI.

## Embeddings and similarity

- Embeddings are generated with `nomic-embed-text` through Ollama when available.
- If Ollama is not available, the processor and query engine fall back to dummy embeddings.
- Similarity is computed with cosine similarity over embeddings in `chatBIS.query.query`.

## Retrieval size

- The multi-agent RAG path retrieves the top 3 chunks per query.
- The standalone `RAGQueryEngine.query()` default is `top_k=5`.

These values are not exposed in the CLI and must be changed in code if you need different retrieval sizes.
