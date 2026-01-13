This how-to shows how to run the Flask web server with custom settings.

# Run web server

```bash
python -m chatBIS web --data .\data\processed --host 127.0.0.1 --port 5000
```

Enable debug mode while developing:

```bash
python -m chatBIS web --data .\data\processed --debug
```
