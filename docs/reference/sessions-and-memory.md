This reference lists how sessions and memory are stored and managed.

# Sessions and memory

## Session IDs

- A new session ID is generated with UUIDs in `ConversationEngine.create_session()`.
- The CLI prints the session ID when it starts a session.
- Use `--session-id` to resume an existing session.

## Memory storage

- Conversation history is stored in SQLite using LangGraph's `SqliteSaver`.
- The default database path is `<data_dir>\conversation_memory.db`.

## History limits

- The conversation engine keeps the last 20 messages in memory (10 exchanges).
- Older messages are truncated from the in-memory history to keep the session size bounded.

## Clearing sessions

- The CLI `clear` command starts a new session ID.
- The current implementation does not delete the previous history from SQLite; it just switches to a new session.
