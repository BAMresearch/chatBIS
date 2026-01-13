This how-to shows how to run the CLI with an explicit memory database path.

# Run CLI with memory

```bash
python -m chatBIS query --data .\data\processed --memory-db .\data\processed\conversation_memory.db
```

If you omit `--memory-db`, chatBIS stores the memory database at `<data_dir>\conversation_memory.db`.
