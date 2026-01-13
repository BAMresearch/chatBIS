This explanation describes why chatBIS uses a router to choose between RAG and pybis actions.

# Multi-agent routing

The conversation engine uses a router to decide whether a user message should be handled by the RAG agent or by pybis function calling. This keeps documentation questions separate from action requests.

## Decisions

- `rag` for documentation and conceptual questions
- `function_call` for action requests such as listing samples or connecting to openBIS
- `conversation` is a fallback path that routes to RAG

## Routing signals

The router checks keywords and patterns in the user query. Examples include:

- Connection keywords such as `connect`, `login`, and `disconnect` are routed to `function_call`.
- Documentation patterns such as `how to` and `what is` are routed to `rag`.
- If both types of signals are present, the router defaults to documentation unless an action is explicit.

The router logs its decision, but it does not expose chain-of-thought reasoning.
