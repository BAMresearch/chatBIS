This explanation describes why chatBIS integrates pybis and how tool calls are executed.

# PyBIS integration

chatBIS wraps pybis methods in LangChain tool objects so the conversation engine can execute openBIS actions on demand.

## Tool manager

- `PyBISToolManager` builds a catalog of tools such as listing spaces, samples, and datasets.
- Tools are only available when the `pybis` package is installed.
- Each tool parses simple `key=value` input strings to extract parameters.

## Connection handling

- The tool manager tries to auto-connect using `OPENBIS_URL`, `OPENBIS_USERNAME`, and `OPENBIS_PASSWORD`.
- If auto-connect is not configured, the user must connect using a chat request that triggers the `connect_to_openbis` tool.

## Read vs write operations

The tool catalog includes both read and write operations. Be cautious when granting credentials, especially in shared environments.
