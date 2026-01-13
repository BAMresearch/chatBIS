This tutorial will walk you through a first CLI chat session and show how memory behaves.

# First chat session (CLI)

## Start the CLI

If you already have processed data, run:

```bash
python -m chatBIS query --data .\data\processed
```

If you do not have processed data yet, run `python -m chatBIS` once to build it, then come back to the CLI command above.

## Ask a question and a follow-up

Try a documentation question, then a follow-up that relies on context:

```text
You: What is openBIS?
You: And how does it organize experiments?
```

The assistant uses the conversation history within the same session, so follow-up questions are answered with the earlier context in mind.

## Verify memory and session handling

- When the session starts, chatBIS prints a session ID. Save it if you want to resume later.
- Type `clear` to start a new session ID.
- Type `exit` or `quit` to end the session.

To continue a session later, see `howtos/continue-session.md`.
