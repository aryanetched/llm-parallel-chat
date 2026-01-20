# Local Chat Client

A terminal-based chat client that connects to an OpenAI-compatible API endpoint.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Run with default host (tempest03)
python chat.py

# Or specify a different host
EMULATION_HOST=myhost python chat.py
```

## Commands

- `quit` or `exit` - End the session
- `clear` - Clear conversation history

## Configuration

The client connects to `http://${EMULATION_HOST}:4317/v1` by default.

- Default host: `tempest03`
- Port: `4317`
- Health endpoint: `/health`
- Chat endpoint: `/v1/chat/completions`
