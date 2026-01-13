This how-to shows how to continue an existing session using a saved session ID.

# Continue a session

1. Start a session and copy the session ID printed by the CLI.
2. Resume with that ID:

```bash
python -m chatBIS query --data .\data\processed --session-id <your-session-id>
```

The session history is stored in the same SQLite database configured by `--memory-db`.
