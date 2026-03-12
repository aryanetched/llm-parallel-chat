#!/usr/bin/env python3
"""
ChatGPT-like web interface for local LLM inference.
Connects to the same backend as chat.py.
"""

import json
import re
from typing import Any, List, Optional, Dict
from contextlib import asynccontextmanager

from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import httpx
from pydantic import BaseModel

DEFAULT_WS_NUM = "57"
HOST_TEMPLATE = "http://ws{ws_num}-dk.srw.i.etched.com:8000"
TIMEOUT = 120


def make_base_url(ws_num: str) -> str:
    # Only allow alphanumeric to prevent SSRF
    if not re.match(r'^[a-zA-Z0-9]+$', ws_num):
        raise ValueError(f"Invalid ws_num: {ws_num!r}")
    return HOST_TEMPLATE.format(ws_num=ws_num)

http_client: Optional[httpx.AsyncClient] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global http_client
    http_client = httpx.AsyncClient(timeout=TIMEOUT)
    yield
    await http_client.aclose()


app = FastAPI(title="Chat UI", lifespan=lifespan)


class ChatMessage(BaseModel):
    role: str
    text: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    model: str = "llama8b"
    instructions: str = "You are a helpful assistant. Answer in a concise, but friendly manner."
    max_output_tokens: int = 50000
    ws_num: str = DEFAULT_WS_NUM


def format_messages_for_api(messages: List[ChatMessage]) -> List[Dict[str, Any]]:
    return [
        {
            "type": "message",
            "role": msg.role,
            "content": [{"type": "input_text", "text": msg.text}],
        }
        for msg in messages
    ]


@app.get("/health")
async def health(ws: str = Query(default=DEFAULT_WS_NUM)):
    try:
        base_url = make_base_url(ws)
        resp = await http_client.get(f"{base_url}/health", timeout=5)
        llm_ok = resp.status_code == 200
    except Exception:
        llm_ok = False
    return {"status": "ok", "llm_backend": llm_ok, "ws_num": ws}


@app.post("/api/chat")
async def chat(request: ChatRequest):
    base_url = make_base_url(request.ws_num)
    payload = {
        "model": request.model,
        "input": format_messages_for_api(request.messages),
        "instructions": request.instructions,
        "max_output_tokens": request.max_output_tokens,
        "stream": True,
    }

    async def generate():
        try:
            async with http_client.stream(
                "POST",
                f"{base_url}/v1/responses",
                json=payload,
                timeout=TIMEOUT,
            ) as response:
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data = line[6:]
                        if data == "[DONE]":
                            yield f"data: {json.dumps({'done': True})}\n\n"
                            break
                        try:
                            event = json.loads(data)
                            if event.get("type") == "response.output_text.delta":
                                delta = event.get("delta", "")
                                yield f"data: {json.dumps({'delta': delta})}\n\n"
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
    with open("static/chat.html", "r") as f:
        return f.read()


app.mount("/static", StaticFiles(directory="static"), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080, reload=False)
