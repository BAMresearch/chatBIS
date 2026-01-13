This explanation describes why chatBIS persists memory using LangGraph and SQLite.

# Memory and state

chatBIS stores conversation state so the assistant can refer to previous exchanges in a session.

## How memory is stored

- The conversation engine uses a LangGraph `StateGraph` with a `SqliteSaver` checkpointer.
- Each session ID is stored as a separate thread in the SQLite database.
- The default path is `<data_dir>\conversation_memory.db`.

## Session isolation

Messages are scoped by session ID, so different sessions do not share history.

## History truncation

The engine keeps the last 20 messages in memory to cap context size. Older messages are dropped from the in-memory state but remain in SQLite.
