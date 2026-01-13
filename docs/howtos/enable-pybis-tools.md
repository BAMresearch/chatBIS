This how-to shows how to enable pybis function calling and connect to openBIS.

# Enable pybis tools

## Install pybis

`pybis` is already listed in `requirements.txt`. Make sure it is installed:

```bash
pip install -r requirements.txt
```

## Provide credentials safely

chatBIS can auto-connect if these environment variables are set:

- `OPENBIS_URL`
- `OPENBIS_USERNAME`
- `OPENBIS_PASSWORD`

You can place them in a `.env` file (loaded via `python-dotenv`) or export them in your shell before launching chatBIS.

## Connect from the chat

You can also ask chatBIS to connect using a natural language request. For example:

```text
Connect to openBIS at https://demo.openbis.ch with username YOUR_USER and password YOUR_PASSWORD
```

Connection state is stored in memory for the running process. If you restart the app, connect again or use environment variables.
