#!/usr/bin/env python3
"""
Benchmark: spam the LLM endpoint with concurrent streaming requests.
Measures TTFT, throughput, latency percentiles, and more.
"""

import argparse
import asyncio
import json
import random
import time
from dataclasses import dataclass, field

import httpx
from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, MofNCompleteColumn
import numpy as np

LLM_BASE_URL = "http://ws17-dk.srw.i.etched.com:8000"
MODEL = "llama8b"
INSTRUCTIONS = "You are a helpful assistant. Provide thorough, detailed responses."

PROMPT_TEMPLATES = [
    (
        "You are a historian specializing in {topic}. A student has asked you the following question "
        "and expects a comprehensive, well-sourced answer with specific dates, names, and events.\n\n"
        "Context: The student is writing a research paper on {topic} and needs detailed primary and "
        "secondary source analysis. They are particularly interested in the causal relationships "
        "between events, the key figures involved, and the long-term consequences.\n\n"
        "Question: {question}\n\n"
        "Please provide a thorough response covering multiple perspectives and time periods."
    ),
    (
        "You are a senior software engineer conducting a technical deep-dive on {topic}. "
        "Your audience is a team of experienced developers who need actionable, detailed guidance.\n\n"
        "Background: The team has been working on a large-scale distributed system and encountered "
        "challenges related to {topic}. They need to understand the trade-offs between different "
        "approaches, common pitfalls, and industry best practices.\n\n"
        "Question: {question}\n\n"
        "Provide a detailed technical analysis with concrete examples, code patterns where relevant, "
        "and performance considerations."
    ),
    (
        "You are a research scientist explaining {topic} to a graduate-level audience. "
        "The explanation should be rigorous but accessible, covering both theoretical foundations "
        "and practical applications.\n\n"
        "Context: This is for a seminar series where participants have strong quantitative backgrounds "
        "but may not be specialists in {topic}. They want to understand the current state of the art, "
        "key open problems, and how this field connects to adjacent disciplines.\n\n"
        "Question: {question}\n\n"
        "Include relevant mathematical intuitions, landmark papers, and real-world case studies."
    ),
    (
        "You are an economics professor preparing a lecture on {topic}. Your students are advanced "
        "undergraduates who are comfortable with quantitative analysis and want to understand "
        "both the theoretical models and empirical evidence.\n\n"
        "Context: This lecture is part of a series examining how {topic} has shaped modern economic "
        "policy. Students have already studied basic micro and macroeconomic theory and are ready "
        "for nuanced, data-driven discussion.\n\n"
        "Question: {question}\n\n"
        "Cover the major schools of thought, empirical findings, and policy implications."
    ),
    (
        "You are a philosophy professor leading a seminar on {topic}. The participants are doctoral "
        "students who expect precise argumentation, engagement with primary texts, and critical "
        "analysis of competing positions.\n\n"
        "Context: This seminar explores the intersection of {topic} with contemporary debates in "
        "ethics, epistemology, and political philosophy. Students have read foundational texts "
        "and are looking for deeper synthesis and original analysis.\n\n"
        "Question: {question}\n\n"
        "Present the strongest versions of opposing arguments and evaluate them rigorously."
    ),
]

TOPIC_QUESTION_PAIRS = [
    ("the Roman Empire's decline", "What were the primary internal and external factors that led to the fall of the Western Roman Empire, and how did they interact?"),
    ("medieval Islamic science", "How did the Islamic Golden Age preserve and advance Greek scientific knowledge, and what were its most lasting contributions?"),
    ("the French Revolution", "What were the socioeconomic conditions that made the French Revolution inevitable, and how did it reshape European politics?"),
    ("the Industrial Revolution", "How did the transition from agrarian to industrial economies transform social structures, labor, and urbanization in 19th-century Britain?"),
    ("the Cold War's technological race", "How did geopolitical competition between the US and USSR drive technological innovation in space, computing, and nuclear energy?"),
    ("database scaling strategies", "What are the fundamental trade-offs between horizontal and vertical scaling for relational databases under high write throughput?"),
    ("consensus algorithms in distributed systems", "Compare Raft, Paxos, and PBFT in terms of fault tolerance, performance, and practical deployment considerations."),
    ("memory management in systems programming", "How do modern garbage collectors compare to manual memory management in terms of latency, throughput, and safety guarantees?"),
    ("neural network optimization", "What are the key challenges in training very deep neural networks, and how do techniques like batch normalization and residual connections address them?"),
    ("compiler design and optimization", "How do modern optimizing compilers perform loop vectorization, and what are the limits of automatic parallelization?"),
    ("quantum computing fundamentals", "Explain the principles of quantum error correction and why it is essential for building practical quantum computers."),
    ("evolutionary biology and adaptation", "How does natural selection operate at the molecular level, and what role do genetic drift and neutral mutations play in evolution?"),
    ("climate modeling and prediction", "What are the primary sources of uncertainty in modern climate models, and how do ensemble methods help address them?"),
    ("behavioral economics and decision-making", "How do cognitive biases systematically distort human economic decision-making, and what are the implications for market efficiency?"),
    ("the ethics of artificial intelligence", "What are the strongest arguments for and against granting legal personhood to advanced AI systems?"),
    ("Renaissance art and patronage", "How did the Medici family's patronage system shape the artistic output of the Italian Renaissance?"),
    ("the Silk Road's cultural impact", "How did trade along the Silk Road facilitate the exchange of religious ideas, technologies, and artistic styles between civilizations?"),
    ("cryptographic protocol design", "What are the security assumptions underlying modern TLS, and how would quantum computing affect them?"),
    ("operating system kernel design", "Compare monolithic kernels and microkernels in terms of performance, security, and maintainability with specific real-world examples."),
    ("the philosophy of consciousness", "What are the strongest arguments for and against physicalism as an explanation for conscious experience?"),
    ("supply chain optimization", "How do modern supply chain networks use predictive analytics and real-time data to minimize disruption and optimize inventory?"),
    ("the history of written language", "How did the development of writing systems in Mesopotamia, Egypt, and China differ, and what drove their independent invention?"),
    ("protein folding and drug discovery", "How has computational protein structure prediction advanced drug discovery, and what are the remaining challenges?"),
    ("game theory in international relations", "How do game-theoretic models explain nuclear deterrence, and where do they fail to capture real-world dynamics?"),
    ("the neuroscience of learning and memory", "What are the molecular and synaptic mechanisms underlying long-term potentiation, and how do they relate to learning?"),
]


def generate_prompt(request_id: int) -> str:
    """Generate a unique prompt for each request to defeat prefix caching."""
    template = PROMPT_TEMPLATES[request_id % len(PROMPT_TEMPLATES)]
    topic, question = TOPIC_QUESTION_PAIRS[request_id % len(TOPIC_QUESTION_PAIRS)]
    seed_phrase = f"[Request #{request_id}, seed={random.randint(100000, 999999)}] "
    return seed_phrase + template.format(topic=topic, question=question)


@dataclass
class RequestResult:
    request_id: int
    ttft: float = 0.0            # seconds
    total_time: float = 0.0      # seconds
    output_tokens: int = 0
    input_tokens: int = 0
    output_text: str = ""
    error: str = ""
    first_token_arrived: bool = False
    tpot: float = 0.0            # time per output token (excluding first)


@dataclass
class BenchmarkStats:
    results: list[RequestResult] = field(default_factory=list)
    wall_start: float = 0.0
    wall_end: float = 0.0

    @property
    def successful(self) -> list[RequestResult]:
        return [r for r in self.results if not r.error]

    @property
    def failed(self) -> list[RequestResult]:
        return [r for r in self.results if r.error]


@dataclass
class LiveCounters:
    """Shared mutable counters updated by all request tasks."""
    total_tokens: int = 0
    completed: int = 0
    failed: int = 0
    first_token_count: int = 0
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


async def send_request(
    client: httpx.AsyncClient,
    request_id: int,
    max_tokens: int,
    timeout: float,
    semaphore: asyncio.Semaphore,
    counters: LiveCounters,
) -> RequestResult:
    result = RequestResult(request_id=request_id)
    prompt = generate_prompt(request_id)

    payload = {
        "model": MODEL,
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": prompt}],
            }
        ],
        "instructions": INSTRUCTIONS,
        "max_output_tokens": max_tokens,
        "stream": True,
    }

    async with semaphore:
        start = time.perf_counter()
        try:
            async with client.stream(
                "POST",
                f"{LLM_BASE_URL}/v1/responses",
                json=payload,
                timeout=timeout,
            ) as response:
                delta_count = 0
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data = line[6:]
                    try:
                        event = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    etype = event.get("type", "")
                    if etype == "response.output_text.delta":
                        now = time.perf_counter()
                        if not result.first_token_arrived:
                            result.ttft = now - start
                            result.first_token_arrived = True
                            counters.first_token_count += 1
                        result.output_text += event.get("delta", "")
                        delta_count += 1
                        counters.total_tokens += 1
                    elif etype == "response.completed":
                        usage = event.get("response", {}).get("usage", {})
                        result.input_tokens = usage.get("input_tokens", 0)
                        result.output_tokens = usage.get("output_tokens", 0)
                        break

                result.total_time = time.perf_counter() - start
                if result.output_tokens == 0:
                    result.output_tokens = delta_count
                if result.output_tokens > 1 and result.ttft > 0:
                    result.tpot = (result.total_time - result.ttft) / (result.output_tokens - 1)
                counters.completed += 1

        except Exception as e:
            result.total_time = time.perf_counter() - start
            result.error = str(e)
            counters.failed += 1

    return result


def print_results(console: Console, stats: BenchmarkStats, num_requests: int, max_tokens: int, concurrency: int):
    ok = stats.successful
    fail = stats.failed
    wall_time = stats.wall_end - stats.wall_start

    console.print()
    console.print(Panel.fit(
        f"[bold]Benchmark Complete[/bold]\n"
        f"[dim]{num_requests} requests | {max_tokens} max tokens | concurrency {concurrency}[/dim]",
        border_style="green",
    ))

    summary = Table(title="Summary", show_header=False, border_style="cyan")
    summary.add_column("Metric", style="bold")
    summary.add_column("Value", justify="right")
    summary.add_row("Total requests", str(num_requests))
    summary.add_row("Successful", f"[green]{len(ok)}[/green]")
    summary.add_row("Failed", f"[red]{len(fail)}[/red]" if fail else "[green]0[/green]")
    if ok:
        avg_input = sum(r.input_tokens for r in ok) / len(ok)
        summary.add_row("Avg input tokens / request", f"{avg_input:.0f}")
    summary.add_row("Wall-clock time", f"{wall_time:.2f}s")

    if ok:
        total_tokens = sum(r.output_tokens for r in ok)
        total_gen_time = sum(r.total_time for r in ok)
        avg_ttft = np.mean([r.ttft for r in ok])
        avg_total = np.mean([r.total_time for r in ok])
        avg_decode = avg_total - avg_ttft
        prefill_decode_ratio = avg_ttft / avg_decode if avg_decode > 0 else float("inf")
        summary.add_row("Total output tokens", f"{total_tokens:,}")
        summary.add_row("Aggregate throughput", f"[bold cyan]{total_tokens / wall_time:.1f}[/bold cyan] tok/s")
        summary.add_row("Per-request avg throughput", f"{total_tokens / total_gen_time * len(ok) / len(ok):.1f} tok/s")
        summary.add_row("Avg prefill (TTFT)", f"{avg_ttft:.3f}s")
        summary.add_row("Avg decode", f"{avg_decode:.3f}s")
        summary.add_row("Prefill / Decode ratio", f"[bold yellow]{prefill_decode_ratio:.3f}[/bold yellow]")

    console.print(summary)

    if not ok:
        console.print("[red]No successful requests to report on.[/red]")
        if fail:
            for r in fail[:5]:
                console.print(f"  [red]#{r.request_id}: {r.error[:120]}[/red]")
        return

    ttfts = np.array([r.ttft for r in ok])
    totals = np.array([r.total_time for r in ok])
    tokens = np.array([r.output_tokens for r in ok])
    per_req_tps = np.array([r.output_tokens / r.total_time if r.total_time > 0 else 0 for r in ok])
    tpots = np.array([r.tpot for r in ok if r.tpot > 0])

    def pct_row(table: Table, label: str, arr: np.ndarray, fmt: str = ".3f", unit: str = "s"):
        table.add_row(
            label,
            f"{np.min(arr):{fmt}}{unit}",
            f"{np.percentile(arr, 25):{fmt}}{unit}",
            f"{np.median(arr):{fmt}}{unit}",
            f"{np.mean(arr):{fmt}}{unit}",
            f"{np.percentile(arr, 75):{fmt}}{unit}",
            f"{np.percentile(arr, 90):{fmt}}{unit}",
            f"{np.percentile(arr, 99):{fmt}}{unit}",
            f"{np.max(arr):{fmt}}{unit}",
        )

    lat = Table(title="Latency Percentiles", border_style="yellow")
    for col in ["Metric", "Min", "P25", "P50 (med)", "Mean", "P75", "P90", "P99", "Max"]:
        lat.add_column(col, justify="right" if col != "Metric" else "left")
    pct_row(lat, "TTFT", ttfts)
    pct_row(lat, "Total latency", totals)
    if len(tpots) > 0:
        pct_row(lat, "Inter-token lat", tpots * 1000, fmt=".2f", unit="ms")
    pct_row(lat, "Tokens generated", tokens, fmt=".0f", unit="")
    pct_row(lat, "Per-req tok/s", per_req_tps, fmt=".1f", unit="")
    console.print(lat)

    if fail:
        err_table = Table(title="Errors (first 10)", border_style="red")
        err_table.add_column("Req #", justify="right")
        err_table.add_column("Error")
        for r in fail[:10]:
            err_table.add_row(str(r.request_id), r.error[:150])
        console.print(err_table)

    console.print()
    console.print(f"[dim]First response sample (req #0, {ok[0].output_tokens} tokens):[/dim]")
    console.print(Panel(ok[0].output_text[:500] + ("..." if len(ok[0].output_text) > 500 else ""), border_style="dim"))


async def check_health(console: Console) -> bool:
    """Check if the LLM endpoint is reachable before benchmarking."""
    console.print(f"[dim]Checking health at {LLM_BASE_URL}/health ...[/dim]", end=" ")
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(f"{LLM_BASE_URL}/health")
            if resp.status_code == 200:
                console.print("[green]✓ Healthy[/green]")
                return True
            console.print(f"[red]✗ Status {resp.status_code}[/red]")
    except Exception as e:
        console.print(f"[red]✗ {e}[/red]")
    return False


async def run_benchmark(num_requests: int, max_tokens: int, concurrency: int, timeout: float):
    console = Console()
    sample_prompt = generate_prompt(0)
    approx_input_toks = len(sample_prompt.split()) * 4 // 3
    console.print(Panel.fit(
        f"[bold yellow]LLM Benchmark[/bold yellow]\n"
        f"[dim]Target: {LLM_BASE_URL}  |  Model: {MODEL}[/dim]\n"
        f"[dim]{num_requests} requests  |  ~{approx_input_toks} input tokens  |  {max_tokens} max output tokens  |  concurrency {concurrency}[/dim]\n"
        f"[dim]{len(TOPIC_QUESTION_PAIRS)} unique prompts (no prefix sharing)[/dim]",
        border_style="yellow",
    ))

    if not await check_health(console):
        console.print("[red bold]Aborting: LLM endpoint is not reachable.[/red bold]")
        return

    limits = httpx.Limits(
        max_connections=concurrency + 50,
        max_keepalive_connections=concurrency + 50,
    )
    client = httpx.AsyncClient(limits=limits, timeout=timeout)
    semaphore = asyncio.Semaphore(concurrency)
    stats = BenchmarkStats()
    counters = LiveCounters()

    def build_live_table() -> Table:
        elapsed = time.perf_counter() - stats.wall_start if stats.wall_start else 0
        tps = counters.total_tokens / elapsed if elapsed > 0 else 0
        t = Table(border_style="blue", show_header=False, padding=(0, 1))
        t.add_column("Metric", style="bold")
        t.add_column("Value", justify="right", min_width=12)
        t.add_row("Elapsed", f"{elapsed:.1f}s")
        t.add_row("Completed", f"[green]{counters.completed}[/green] / {num_requests}")
        t.add_row("Failed", f"[red]{counters.failed}[/red]" if counters.failed else "0")
        t.add_row("Got first token", f"[cyan]{counters.first_token_count}[/cyan] / {num_requests}")
        t.add_row("Tokens received", f"[bold]{counters.total_tokens:,}[/bold]")
        t.add_row("Live throughput", f"[bold cyan]{tps:,.0f} tok/s[/bold cyan]")
        return t

    stats.wall_start = time.perf_counter()
    tasks = [
        send_request(client, i, max_tokens, timeout, semaphore, counters)
        for i in range(num_requests)
    ]
    gather_task = asyncio.ensure_future(asyncio.gather(*tasks))

    with Live(build_live_table(), console=console, refresh_per_second=4) as live:
        while not gather_task.done():
            live.update(build_live_table())
            await asyncio.sleep(0.25)
        live.update(build_live_table())

    stats.results = gather_task.result()
    stats.wall_end = time.perf_counter()

    await client.aclose()
    print_results(console, stats, num_requests, max_tokens, concurrency)


def main():
    parser = argparse.ArgumentParser(description="Benchmark LLM endpoint with concurrent streaming requests")
    parser.add_argument("-n", "--num-requests", type=int, default=1000, help="Number of requests (default: 1000)")
    parser.add_argument("-t", "--max-tokens", type=int, default=1000, help="Max output tokens per request (default: 1000)")
    parser.add_argument("-c", "--concurrency", type=int, default=1000, help="Max concurrent requests (default: 1000)")
    parser.add_argument("--timeout", type=float, default=300, help="Per-request timeout in seconds (default: 300)")
    parser.add_argument("--url", type=str, default=None, help="Override LLM base URL")
    parser.add_argument("--model", type=str, default=None, help="Override model name")
    args = parser.parse_args()

    if args.url:
        global LLM_BASE_URL
        LLM_BASE_URL = args.url
    if args.model:
        global MODEL
        MODEL = args.model

    asyncio.run(run_benchmark(args.num_requests, args.max_tokens, args.concurrency, args.timeout))


if __name__ == "__main__":
    main()
