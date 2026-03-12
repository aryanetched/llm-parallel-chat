#!/usr/bin/env python3
"""
TTFT Sweep: measure Time-To-First-Token across input token sizes.

Sends streaming requests with padded prompts from ~1k to ~50k input tokens,
skipping some points in the middle. Prints a table of TTFT vs input size.
"""

import argparse
import asyncio
import json
import statistics
import time

import httpx
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from datetime import datetime

#BASE_URL = "http://ws17-dk.srw.i.etched.com:8000"
BASE_URL="http://sd-setup-09.srw.i.etched.com:8000"
MODEL = "llama8b"

# Token sweep: dense at the low end, sparser in the middle/top.
# 512-token steps from 1k→8k, then jumps to 50k.
TOKEN_SIZES = (
    list(range(1024, 8193, 512))          # 1k–8k every 512
    + [10240, 12288, 16384, 20480, 24576, 32768, 40960, 49152]  # 10k–48k
)
# TOKEN_SIZES=[1024 * 8 * 2]
FILLER = (
    "xxx The following is filler text used to pad the prompt to a specific length. "
    "xxx It contains no useful information and should be ignored. "
) * 200  # repeat so we can slice cheaply

import random
import string
import time

def random_words():
    random.seed(time.time_ns())
    return " ".join(
        "".join(random.choices(string.ascii_lowercase, k=4))
        for _ in range(8)
    )


def make_prompt(target_tokens: int) -> str:
    """Build a prompt that approximates target_tokens input tokens.

    Uses ~4 chars per token heuristic. The actual question is appended at
    the end so the model always has something to answer.
    """
    question = "In one sentence, what is 2 + 2?"
    # Reserve ~20 tokens for the question and instructions overhead.
    pad_tokens = max(0, target_tokens - 30)
    pad_chars = pad_tokens * 4
    # Build filler that's long enough
    filler = (FILLER * ((pad_chars // len(FILLER)) + 2))[:pad_chars]
    return f"{filler}\n\n{question}"


async def measure_ttft(
    client: httpx.AsyncClient,
    prompt: str,
    timeout: float,
) -> float | None:
    """Return TTFT in seconds, or None on error."""
    payload = {
        "model": MODEL,
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": prompt}],
            }
        ],
        "instructions": "You are a helpful assistant.",
        "max_output_tokens": 1,   # tiny output — we only care about TTFT
        "stream": True,
    }

    start = time.perf_counter()
    try:
        async with client.stream(
            "POST",
            f"{BASE_URL}/v1/responses",
            json=payload,
            timeout=timeout,
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                try:
                    event = json.loads(line[6:])
                except json.JSONDecodeError:
                    continue
                if event.get("type") == "response.output_text.delta":
                    return time.perf_counter() - start
    except Exception as e:
        print(f"  [error] {e}")
        return None
    return None


async def sweep(samples: int, timeout: float, csv_out: str | None):
    console = Console()
    console.print(Panel.fit(
        f"[bold yellow]TTFT Sweep[/bold yellow]\n"
        f"[dim]{len(TOKEN_SIZES)} token sizes  |  {samples} sample(s) each  |  {MODEL} @ {BASE_URL}[/dim]",
        border_style="yellow",
    ))

    # Health check
    console.print("[dim]Checking health...[/dim]", end=" ")
    try:
        async with httpx.AsyncClient(timeout=10) as hc:
            r = await hc.get(f"{BASE_URL}/health")
            if r.status_code == 200:
                console.print("[green]✓[/green]")
            else:
                console.print(f"[red]✗ {r.status_code}[/red]")
                return
    except Exception as e:
        console.print(f"[red]✗ {e}[/red]")
        return

    limits = httpx.Limits(max_connections=4, max_keepalive_connections=4)
    client = httpx.AsyncClient(limits=limits, timeout=timeout)

    rows: list[tuple[int, float, float, float, list[float]]] = []  # (toks, min, median, max, raw)

    for toks in TOKEN_SIZES:
        prompt = make_prompt(toks)
        approx_chars = len(prompt)
        approx_toks = approx_chars // 4

        ttfts: list[float] = []
        console.print(
            f"  [cyan]~{approx_toks:,} tok[/cyan] ({approx_chars:,} chars) … ",
            end="",
        )
        for _ in range(samples):
            prompt = make_prompt(toks)
            t = await measure_ttft(client, prompt, timeout)
            if t is not None:
                print(f"{datetime.now()} ran in {t}")
                ttfts.append(t)

        if ttfts:
            mn = min(ttfts)
            med = statistics.median(ttfts)
            mx = max(ttfts)
            console.print(f"min={mn:.3f}s  med={med:.3f}s  max={mx:.3f}s")
            rows.append((approx_toks, mn, med, mx, ttfts))
        else:
            console.print("[red]all failed[/red]")
            rows.append((approx_toks, float("nan"), float("nan"), float("nan"), []))

    await client.aclose()

    # Summary table
    table = Table(title="TTFT vs Input Token Count", border_style="cyan")
    table.add_column("~Input Tokens", justify="right", style="bold")
    table.add_column("Min TTFT", justify="right")
    table.add_column("Median TTFT", justify="right")
    table.add_column("Max TTFT", justify="right")
    if samples > 1:
        table.add_column("Samples", justify="right")

    for approx_toks, mn, med, mx, raw in rows:
        if raw:
            row = [
                f"{approx_toks:,}",
                f"{mn:.3f}s",
                f"[bold]{med:.3f}s[/bold]",
                f"{mx:.3f}s",
            ]
        else:
            row = [f"{approx_toks:,}", "[red]err[/red]", "[red]err[/red]", "[red]err[/red]"]
        if samples > 1:
            row.append(str(len(raw)))
        table.add_row(*row)

    console.print()
    console.print(table)

    if csv_out:
        with open(csv_out, "w") as f:
            f.write("approx_input_tokens,min_ttft_s,median_ttft_s,max_ttft_s,raw_ttfts\n")
            for approx_toks, mn, med, mx, raw in rows:
                raw_str = ";".join(f"{v:.4f}" for v in raw)
                f.write(f"{approx_toks},{mn:.4f},{med:.4f},{mx:.4f},{raw_str}\n")
        console.print(f"[dim]Results saved to {csv_out}[/dim]")


def main():
    parser = argparse.ArgumentParser(description="Sweep TTFT across input token sizes")
    parser.add_argument("-s", "--samples", type=int, default=3,
                        help="Samples per token size (default: 3)")
    parser.add_argument("--timeout", type=float, default=120,
                        help="Per-request timeout in seconds (default: 120)")
    parser.add_argument("--csv", type=str, default=None,
                        help="Write results to CSV file")
    parser.add_argument("--url", type=str, default=None, help="Override base URL")
    parser.add_argument("--model", type=str, default=None, help="Override model name")
    args = parser.parse_args()

    if args.url:
        global BASE_URL
        BASE_URL = args.url
    if args.model:
        global MODEL
        MODEL = args.model

    asyncio.run(sweep(args.samples, args.timeout, args.csv))


if __name__ == "__main__":
    main()
