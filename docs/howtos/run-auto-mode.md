This how-to shows how to run the automatic pipeline mode.

# Run auto mode

Auto mode checks for processed data and runs the full pipeline if needed.

```bash
python -m chatBIS
```

You can also call the hidden subcommand directly:

```bash
python -m chatBIS auto
```

Auto mode checks for `data/processed/chunks.json` and runs `scrape`, `process`, and `query` if it is missing.
