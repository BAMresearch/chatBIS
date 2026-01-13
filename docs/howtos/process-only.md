This how-to shows how to process scraped content into chunks and embeddings.

# Process only

```bash
python -m chatBIS process --input .\data\raw --output .\data\processed
```

Adjust chunking if needed:

```bash
python -m chatBIS process --input .\data\raw --output .\data\processed --min-chunk-size 100 --max-chunk-size 1000 --chunk-overlap 50
```
