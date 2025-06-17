![Python](https://img.shields.io/badge/python-3.8+-yellow) [![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE) 	![GitHub Repo stars](https://img.shields.io/github/stars/BAMresearch/chatBIS) ![Last Commit](https://img.shields.io/github/last-commit/BAMresearch/chatBIS) 

# chatBIS

A RAG-based chatbot with memory for the openBIS documentation, powered by LangGraph and Ollama.

## Overview

This project provides an intelligent chatbot that can answer questions about openBIS using Retrieval Augmented Generation (RAG) with conversation memory. The chatbot remembers previous interactions within a session and provides contextually aware responses.

### Key Features

- **RAG-powered responses**: Uses openBIS documentation for accurate, up-to-date answers
- **Conversation memory**: Remembers user names, previous questions, and context using LangGraph
- **Session management**: Maintains separate conversations with unique session IDs
- **Clean responses**: Filters out internal reasoning for user-friendly output
- **Multi-interface**: Available as both CLI and web interface
- **Persistent storage**: Conversation history stored in SQLite database

### Components

1. **Scraper**: Scrapes content from the openBIS documentation website
2. **Processor**: Processes the scraped content for use in RAG
3. **Conversation Engine**: LangGraph-based engine with memory and RAG integration
4. **Web Interface**: Browser-based chat interface with session management
5. **CLI Interface**: Command-line chat interface with memory

## Installation

### Requirements

- Python 3.8 or higher
- [Ollama](https://ollama.ai/) with the following models:
  - `nomic-embed-text` (for embeddings)
  - `qwen3` (for chat)

### Dependencies

The project uses the following key dependencies:
- **LangGraph**: For conversation flow and memory management
- **LangChain**: For LLM integration and message handling
- **Flask**: For the web interface
- **SQLite**: For persistent conversation storage
- **Ollama**: For local LLM inference

### From Source (Recommended)

```bash
git clone https://github.com/yourusername/openbis-chatbot.git
cd openbis-chatbot
pip install -e .
```

This installs the package in development mode, allowing you to make changes to the code and have them reflected immediately.

### Using pip (Not yet available)

```bash
pip install openbis-chatbot
```

Note: This option will be available once the package is published to PyPI.

## Usage

### Simple Usage (Recommended)

The simplest way to use the chatbot is with a single command:

```bash
python -m openbis_chatbot
```

This will:
1. Check if processed data already exists in the `data/processed` directory
2. If it exists, start the chatbot with that data
3. If not, automatically scrape the openBIS documentation, process it, and then start the chatbot

### Advanced Usage (Component-by-Component)

If you need more control, you can still run each component separately:

#### Scraping Content

```bash
python -m openbis_chatbot scrape --url https://openbis.readthedocs.io/en/latest/ --output ./data/raw
```

#### Processing Content

```bash
python -m openbis_chatbot process --input ./data/raw --output ./data/processed
```

#### Running the Chatbot (CLI with Memory)

```bash
python -m openbis_chatbot query --data ./data/processed
```

The CLI now includes conversation memory features:
- Remembers your name and previous questions within a session
- Type `clear` to start a new conversation
- Type `exit` or `quit` to end the session
- Use `--session-id <id>` to continue a previous conversation

#### Running the Web Interface (with Memory)

```bash
python -m openbis_chatbot --web
```

This will start a web server on http://localhost:5000 where you can interact with the chatbot through a browser.

The web interface includes:
- **Session persistence**: Conversations continue across page refreshes
- **Clear chat button**: Start fresh conversations anytime
- **Memory indicators**: See conversation length and token usage in browser console
- **Responsive design**: Works on desktop and mobile devices

Alternatively, you can use the provided script:

```bash
python scripts/run_web.py
```

Or customize the web interface with additional parameters:

```bash
python -m openbis_chatbot.web.cli --data ./data/processed --host 127.0.0.1 --port 5000
```

## Command-Line Options

### Scraper

```
--url URL             The base URL of the ReadtheDocs site
--output OUTPUT       The directory to save the scraped content to
--version VERSION     The specific version to scrape (e.g., 'en/latest')
--delay DELAY         The delay between requests in seconds (default: 0.5)
--max-pages MAX_PAGES The maximum number of pages to scrape
--verbose             Enable verbose logging
```

### Processor

```
--input INPUT         The directory containing the scraped content
--output OUTPUT       The directory to save the processed content to
--min-chunk-size MIN_CHUNK_SIZE
                      The minimum size of a chunk in characters (default: 100)
--max-chunk-size MAX_CHUNK_SIZE
                      The maximum size of a chunk in characters (default: 1000)
--chunk-overlap CHUNK_OVERLAP
                      The overlap between chunks in characters (default: 50)
--verbose             Enable verbose logging
```

### Chatbot (CLI with Memory)

```
--data DATA           The directory containing the processed content
--model MODEL         The Ollama model to use for chat (default: qwen3)
--memory-db PATH      Path to SQLite database for conversation memory
--session-id ID       Session ID to continue a previous conversation
--verbose             Enable verbose logging
```

### Web Interface

```
--data DATA           The directory containing the processed content (default: ./data/processed)
--host HOST           The host to run the web interface on (default: 0.0.0.0)
--port PORT           The port to run the web interface on (default: 5000)
--model MODEL         The Ollama model to use for chat (default: qwen3)
--top-k TOP_K         The number of chunks to retrieve (default: 5)
--debug               Enable debug mode
```

## How It Works

### Scraper

The scraper works by:
1. Starting from the base URL of the openBIS documentation site
2. Downloading the HTML content of each page
3. Extracting links to other pages on the same domain
4. Following those links to scrape more pages
5. Saving the content of each page to a text file

### Processor

The processor works by:
1. Reading the scraped content from text files
2. Chunking the content into smaller pieces
3. Generating embeddings for each chunk using Ollama's embedding model
4. Saving the chunks and their embeddings to JSON and CSV files

### Conversation Engine (LangGraph-based)

The conversation engine works by:
1. **State Management**: Maintains conversation state using LangGraph's StateGraph
2. **Memory Persistence**: Stores conversation history in SQLite using LangGraph checkpoints
3. **RAG Integration**: Retrieves relevant chunks based on user queries
4. **Context Assembly**: Combines conversation history, RAG context, and current query
5. **Response Generation**: Uses Ollama's chat model with full conversation context
6. **Response Cleaning**: Removes internal reasoning tags for clean user output
7. **Session Management**: Maintains separate conversations with unique session IDs

### Memory Features

- **Conversation History**: Remembers both user messages and assistant responses
- **Session Isolation**: Different sessions don't share memory
- **Token Management**: Automatically limits conversation length (20 messages max)
- **Persistent Storage**: Conversations survive application restarts
- **Context Awareness**: Assistant remembers its own previous offers and responses

### Web Interface

The web interface works by:
1. Starting a Flask web server
2. Serving a responsive HTML/CSS/JavaScript chat interface
3. Handling API requests from the frontend
4. Using the query engine to generate responses
5. Returning the responses to the frontend in JSON format

## Project Structure

```
chatBIS/
├── src/chatBIS/          # Main package
│   ├── scraper/                  # Web scraping components
│   ├── processor/                # Content processing components
│   ├── query/                    # Query and conversation engine
│   │   ├── conversation_engine.py # LangGraph-based conversation engine
│   │   ├── query.py              # RAG query engine
│   │   └── cli.py                # CLI interface with memory
│   ├── web/                      # Web interface
│   └── utils/                    # Utility functions
├── tests/                        # Test suite
├── scripts/                      # Utility scripts
├── data/                         # Data directory
│   ├── raw/                      # Scraped content
│   └── processed/                # Processed chunks and embeddings
├── docs/                         # Documentation
│   └── presentations/            # Project presentations
└── requirements.txt              # Python dependencies
```

## Memory and Token Usage

- **Average tokens per exchange**: ~800-900 tokens
- **Memory overhead**: ~100-200 tokens for conversation history
- **RAG context**: ~500-600 tokens per query
- **Conversation limit**: 20 messages (10 exchanges) per session
- **Storage**: SQLite database for persistent conversation history

## Sources

- [**RAG system**](https://github.com/carlosmada22/ragBIS)
- [**Chatbot interface**](https://github.com/carlosmada22/openbis-chatbot)

## License

This project is licensed under the MIT License - see the LICENSE file for details.
