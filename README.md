# LLM Parallel Chat

A chat interface for LLM endpoints with both a web UI and terminal client.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### Web UI

```bash
# Launch the web server (runs on http://localhost:8080)
python server.py

# Or specify a different host
EMULATION_HOST=myhost python server.py
```

Then open http://localhost:8080 in your browser.

### Terminal Client

```bash
# Run with default host (tempest03)
python chat.py

# Or specify a different host
EMULATION_HOST=myhost python chat.py
```

#### Terminal Commands

- `quit` or `exit` - End the session
- `clear` - Clear conversation history

## Configuration

The client connects to `http://${EMULATION_HOST}:4317` by default.

- Default host: `tempest03`
- Port: `4317`
- Health endpoint: `/health`
- Chat endpoint: `/v1/responses`
