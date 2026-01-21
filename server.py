#!/usr/bin/env python3
"""
Web server for parallel chat UI.
Proxies requests to the LLM endpoint at tempest03:4317.
"""

import os
import json
from typing import Any, Optional, List, Dict
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import httpx
from pydantic import BaseModel
import tiktoken

# Configuration
EMULATION_HOST = os.environ.get("EMULATION_HOST", "tempest03")
LLM_BASE_URL = f"http://{EMULATION_HOST}:4317/v1"
TIMEOUT = 120

# Shared async client
http_client: Optional[httpx.AsyncClient] = None

# Tokenizer (approximate - for display purposes only)
tokenizer = tiktoken.get_encoding("cl100k_base")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global http_client
    http_client = httpx.AsyncClient(timeout=TIMEOUT)
    yield
    await http_client.aclose()


app = FastAPI(title="Parallel Chat UI", lifespan=lifespan)


class ChatMessage(BaseModel):
    role: str
    text: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    model: str = "llama8b"
    instructions: str = "You are a helpful assistant. Answer concisely."
    max_output_tokens: int = 500


class TokenizeRequest(BaseModel):
    text: str


def count_tokens(text: str) -> int:
    """Count tokens (approximate)."""
    return len(tokenizer.encode(text))


def format_messages_for_api(messages: List[ChatMessage]) -> List[Dict[str, Any]]:
    """Convert messages to the Responses API format."""
    return [
        {
            "type": "message",
            "role": msg.role,
            "content": [{"type": "input_text", "text": msg.text}],
        }
        for msg in messages
    ]


@app.get("/health")
async def health():
    """Health check - also checks LLM backend."""
    try:
        resp = await http_client.get(f"http://{EMULATION_HOST}:4317/health")
        llm_ok = resp.status_code == 200
    except Exception:
        llm_ok = False
    return {"status": "ok", "llm_backend": llm_ok}


@app.post("/api/tokenize")
async def tokenize(request: TokenizeRequest):
    """Count tokens in text."""
    token_count = count_tokens(request.text)
    return {"tokens": token_count}


@app.post("/api/chat")
async def chat(request: ChatRequest):
    """Stream a chat response from the LLM."""
    
    # Count input tokens
    input_text = " ".join(msg.text for msg in request.messages)
    input_tokens = count_tokens(input_text + request.instructions)
    
    payload = {
        "model": request.model,
        "input": format_messages_for_api(request.messages),
        "instructions": request.instructions,
        "max_output_tokens": request.max_output_tokens,
        "stream": True,
    }

    async def generate():
        output_tokens = 0
        full_response = ""
        
        # Send initial token count
        yield f"data: {json.dumps({'input_tokens': input_tokens, 'output_tokens': 0})}\n\n"
        
        try:
            async with http_client.stream(
                "POST",
                f"{LLM_BASE_URL}/responses",
                json=payload,
                timeout=TIMEOUT,
            ) as response:
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data = line[6:]
                        if data == "[DONE]":
                            # Final token count
                            output_tokens = count_tokens(full_response)
                            yield f"data: {json.dumps({'done': True, 'input_tokens': input_tokens, 'output_tokens': output_tokens})}\n\n"
                            break
                        try:
                            event = json.loads(data)
                            if event.get("type") == "response.output_text.delta":
                                delta = event.get("delta", "")
                                full_response += delta
                                # Update token count every few tokens
                                output_tokens = count_tokens(full_response)
                                yield f"data: {json.dumps({'delta': delta, 'input_tokens': input_tokens, 'output_tokens': output_tokens})}\n\n"
                        except json.JSONDecodeError:
                            pass
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the main HTML page."""
    with open("static/index.html", "r") as f:
        return f.read()


# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
