#!/usr/bin/env python3
"""
Local Chat Client using OpenAI Responses API

Connects to OpenAI-compatible API at tempest03:4317
Uses the Responses API format for inference.
"""

import os
import sys
from typing import Any, Iterator

import httpx
from openai import APIConnectionError, OpenAI
from openai.types.responses import Response
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

# Configuration
EMULATION_HOST = os.environ.get("EMULATION_HOST", "tempest03")
BASE_URL = f"http://{EMULATION_HOST}:4317"

console = Console()


class ChatClient:
    """Chat client using the OpenAI Responses API."""

    def __init__(
        self,
        base_url: str = BASE_URL,
        model: str = "llama8b",
        api_key: str = "ignored",
        timeout: int = 120,
        instructions: str = "You are a helpful assistant. Answer in a concise, but friendly manner.",
        max_output_tokens: int = 500,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_base_url = f"{self.base_url}/v1"
        
        self.client = OpenAI(
            api_key=api_key,
            base_url=self.api_base_url,
            timeout=timeout,
        )
        
        self.model = model
        self.instructions = instructions
        self.max_output_tokens = max_output_tokens
        self.messages: list[dict[str, Any]] = []

    def add_user_message(self, text: str) -> None:
        """Add a user message to the conversation history."""
        self.messages.append({
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": text}],
        })

    def add_assistant_message(self, text: str) -> None:
        """Add an assistant message to the conversation history."""
        self.messages.append({
            "type": "message",
            "role": "assistant",
            "content": [{"type": "input_text", "text": text}],
        })

    def reset(self) -> None:
        """Reset conversation history."""
        self.messages.clear()

    def health(self) -> bool:
        """Check if server is healthy."""
        try:
            response = httpx.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except (httpx.HTTPError, OSError):
            return False

    def stream_turn(self, user_message: str) -> Iterator[str]:
        """Stream a response for the user message."""
        self.add_user_message(user_message)

        params: dict[str, Any] = {
            "model": self.model,
            "input": self.messages,
            "stream": True,
        }

        if self.instructions:
            params["instructions"] = self.instructions
        if self.max_output_tokens:
            params["max_output_tokens"] = self.max_output_tokens

        stream = self.client.responses.create(**params)
        response_text = ""

        for event in stream:
            if hasattr(event, "type") and event.type == "response.output_text.delta":
                delta = event.delta
                response_text += delta
                yield delta

        self.add_assistant_message(response_text)

    def send_turn(self, user_message: str) -> str:
        """Send a message and get a complete response (non-streaming)."""
        self.add_user_message(user_message)

        params: dict[str, Any] = {
            "model": self.model,
            "input": self.messages,
            "stream": False,
        }

        if self.instructions:
            params["instructions"] = self.instructions
        if self.max_output_tokens:
            params["max_output_tokens"] = self.max_output_tokens

        response: Response = self.client.responses.create(**params)
        
        # Extract response text
        response_text = ""
        if response.output:
            output_item = response.output[0]
            content = getattr(output_item, "content", None)
            if content:
                text = getattr(content[0], "text", None)
                if text:
                    response_text = text

        self.add_assistant_message(response_text)
        return response_text

    def close(self):
        """Close the client."""
        self.client.close()


def main():
    console.print(Panel.fit(
        f"[bold green]Local Chat Client[/bold green]\n"
        f"[dim]Connecting to: {BASE_URL}[/dim]",
        border_style="green"
    ))

    client = ChatClient(
        base_url=BASE_URL,
        model="llama8b",
        max_output_tokens=500,
    )

    # Check health
    console.print("[dim]Checking API health...[/dim]", end=" ")
    if client.health():
        console.print("[green]✓ Connected[/green]")
    else:
        console.print("[red]✗ Failed[/red]")
        console.print(f"[red]Server health check failed at {BASE_URL}/health[/red]")
        sys.exit(1)

    console.print("[dim]Commands: 'quit'/'exit' to end, 'clear' to reset history[/dim]")
    console.print()

    while True:
        try:
            user_input = Prompt.ask("[bold yellow]You[/bold yellow]")

            if not user_input.strip():
                continue

            if user_input.lower() in ("quit", "exit"):
                console.print("[dim]Goodbye![/dim]")
                break

            if user_input.lower() == "clear":
                client.reset()
                console.print("[dim]Conversation cleared.[/dim]")
                continue

            console.print("[bold cyan]Assistant:[/bold cyan] ", end="")
            try:
                for chunk in client.stream_turn(user_input):
                    console.print(chunk, end="")
                console.print()
            except (APIConnectionError, httpx.ConnectError, httpx.TimeoutException) as e:
                console.print(f"\n[red]Connection error: {e}[/red]")
                break
            except Exception as e:
                console.print(f"\n[red]Error: {e}[/red]")

            console.print()

        except KeyboardInterrupt:
            console.print("\n[dim]Goodbye![/dim]")
            break
        except EOFError:
            console.print("\n[dim]Goodbye![/dim]")
            break

    client.close()


if __name__ == "__main__":
    main()
