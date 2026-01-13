This how-to shows how to scrape documentation without running the rest of the pipeline.

# Scrape only

```bash
python -m chatBIS scrape --url https://openbis.readthedocs.io/en/latest/ --output .\data\raw
```

Optional flags:

```bash
python -m chatBIS scrape --url https://openbis.readthedocs.io/en/latest/ --output .\data\raw --version en/latest --delay 0.5 --max-pages 100
```
