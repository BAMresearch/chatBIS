This explanation describes why chatBIS uses RAG, memory, and multi-agent routing for openBIS help.

# Overview

chatBIS is designed to answer openBIS questions using local documentation and to keep context across a session. The system combines three ideas:

- RAG over scraped openBIS documentation
- Persistent conversation memory stored in SQLite
- A router that chooses between documentation answers and pybis actions

This lets chatBIS answer documentation questions and also perform openBIS operations when the user explicitly requests them.
