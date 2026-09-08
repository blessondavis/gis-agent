"""One long-lived MCP server subprocess, owned by a dedicated task.

Two constraints shape this module.

1. The model weights take ~2 GB of VRAM once resident and there is room for
   exactly one copy on a single 8 GB card, so the API process must never load
   the model itself. It talks to this session instead. A side benefit is that
   the agent and the deterministic pipeline drive the *same* tools, so their
   numbers are directly comparable.

2. ``stdio_client`` builds an anyio task group, and an anyio task group belongs
   to the task that opened it. Opening the session inside one HTTP request and
   then using it from a later background task tears the streams down as soon as
   the opening task finishes, which surfaces as "Connection closed" partway
   through the first slow call. So the session is owned by one dedicated worker
   task for the lifetime of the process, and every caller hands work to that
   task through a queue rather than touching the session directly.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Any

_worker: "_McpWorker | None" = None
_worker_lock = asyncio.Lock()


def _server_params():
    from mcp import StdioServerParameters

    env = dict(os.environ)
    # Every real file is cached locally. Staying offline avoids the Hub
    # answering 401 for optional files that do not exist in a gated repo,
    # which otherwise looks exactly like an auth failure.
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    env.setdefault("TRANSFORMERS_VERBOSITY", "error")
    return StdioServerParameters(
        command=sys.executable,
        args=["-m", "gisagent.mcp_servers.roads_server"],
        env=env,
    )


@dataclass
class _Job:
    kind: str                      # "call" | "list"
    name: str = ""
    args: dict = field(default_factory=dict)
    timeout: float | None = 1800
    future: asyncio.Future | None = None


class _McpWorker:
    """Owns the MCP session inside a single task and serialises access to it."""

    def __init__(self) -> None:
        self.queue: asyncio.Queue[_Job] = asyncio.Queue()
        self.ready: asyncio.Future = asyncio.get_event_loop().create_future()
        self.task: asyncio.Task | None = None
        self.tools: list | None = None

    async def start(self) -> None:
        self.task = asyncio.create_task(self._run(), name="mcp-worker")
        await self.ready       # propagates a startup failure to the caller

    async def _run(self) -> None:
        from mcp import ClientSession
        from mcp.client.stdio import stdio_client

        log = open("mcp_server.log", "a", encoding="utf-8", errors="replace")
        try:
            # Everything below stays inside this one task, which is the point.
            async with stdio_client(_server_params(), errlog=log) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    self.tools = (await session.list_tools()).tools
                    if not self.ready.done():
                        self.ready.set_result(True)
                    await self._serve(session)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if not self.ready.done():
                self.ready.set_exception(exc)
            else:
                self._fail_pending(exc)
        finally:
            log.close()

    async def _serve(self, session) -> None:
        while True:
            job = await self.queue.get()
            if job.future is None or job.future.cancelled():
                continue
            try:
                if job.kind == "list":
                    self.tools = (await session.list_tools()).tools
                    job.future.set_result(self.tools)
                else:
                    result = await session.call_tool(
                        job.name, job.args, read_timeout_seconds=job.timeout
                    )
                    job.future.set_result(_parse(result))
            except asyncio.CancelledError:
                if not job.future.done():
                    job.future.cancel()
                raise
            except Exception as exc:
                if not job.future.done():
                    job.future.set_exception(exc)

    def _fail_pending(self, exc: Exception) -> None:
        while not self.queue.empty():
            job = self.queue.get_nowait()
            if job.future and not job.future.done():
                job.future.set_exception(exc)

    async def submit(self, job: _Job):
        job.future = asyncio.get_event_loop().create_future()
        await self.queue.put(job)
        return await job.future

    async def stop(self) -> None:
        if self.task and not self.task.done():
            self.task.cancel()
            try:
                await self.task
            except (asyncio.CancelledError, Exception):
                pass


def _parse(result) -> Any:
    """Flatten a CallToolResult into parsed JSON, or text if it is not JSON."""
    if getattr(result, "structured_content", None):
        return result.structured_content
    parts = []
    for block in getattr(result, "content", []) or []:
        text = getattr(block, "text", None)
        if text:
            parts.append(text)
    text = "\n".join(parts) or "(no output)"
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


async def _get_worker() -> _McpWorker:
    global _worker
    async with _worker_lock:
        if _worker is not None and _worker.task and not _worker.task.done():
            return _worker
        worker = _McpWorker()
        await worker.start()
        _worker = worker
        return _worker


async def list_tools(refresh: bool = False) -> list:
    worker = await _get_worker()
    if worker.tools is not None and not refresh:
        return worker.tools
    return await worker.submit(_Job(kind="list"))


async def call(name: str, args: dict | None = None,
               timeout: float | None = 1800) -> Any:
    worker = await _get_worker()
    return await worker.submit(
        _Job(kind="call", name=name, args=args or {}, timeout=timeout)
    )


async def shutdown() -> None:
    global _worker
    if _worker is not None:
        await _worker.stop()
        _worker = None
