This tutorial will demonstrate how chatBIS routes between documentation answers and pybis actions.

# Multi-agent demo

## Start the CLI with processed data

```bash
python -m chatBIS query --data .\data\processed
```

## Ask two different kinds of questions

1. Documentation style question (RAG routing):

```text
You: What is openBIS?
```

2. Action style question (function routing):

```text
You: List samples in space LAB
```

If you are not connected to openBIS, the action-style request will return a connection error. See `howtos/enable-pybis-tools.md` for connection setup.
