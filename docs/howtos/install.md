This how-to shows how to install chatBIS locally from source.

# Install locally

## Create a virtual environment

```bash
python -m venv .venv
.venv\Scripts\activate
```

## Install dependencies

```bash
pip install -r requirements.txt
pip install -e .
```

## Note on package and module names

The distribution name in `setup.py` is `openbis-chatbot`, but the import and entrypoint module is `chatBIS`. Use `python -m chatBIS` to run the application.
