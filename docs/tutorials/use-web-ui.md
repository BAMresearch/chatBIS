This tutorial will show how to run the web UI and verify session persistence in the browser.

# Use the web UI

## Start the server

From the repository root:

```bash
python -m chatBIS --web
```

This runs the web server on `http://localhost:5000` and uses `data/processed` by default.

## Open the UI

Open your browser to `http://localhost:5000` and start chatting.

## Check session persistence

- The UI stores the session ID in `localStorage`, so refreshes keep the same session.
- Click the trash icon to clear the chat history and reset the session ID.
