"""The agent: a conversational harness over the MCP road tools.

The user talks to it in plain language ("the top-left corner is missing roads,
redo it"), and it decides which tools to call. Conversation state is persisted
per job so a session survives a page reload and the agent remembers what it
already tried.

Why an agent rather than a fixed script: measured on this dataset, the text
prompt alone swings F1 by roughly 5x ("street" scores 0.00 where "road network"
scores 0.48), the best threshold varies per region, and a human looking at the
map is the fastest way to find the parts that came out wrong. That is a search
problem with a measurable objective and a human in the loop.
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from gisagent import mcp_client
from gisagent.config import get_settings

SYSTEM_PROMPT = """\
You are a GIS annotation agent that extracts road networks from satellite
imagery. You work by calling tools, and you improve results by measuring them.

You are talking to a person who is looking at a map of your output. Take their
corrections literally and act on them with tools; do not just agree.

Imagery is 1 m/pixel aerial photography with ground-truth road masks available,
so you can score your own work.

Typical first run for a job:
  tile_region -> segment_chips -> stitch_result -> evaluate_result -> improve
  -> vectorize_result

What is known about this model on this imagery, from measurement:
  - The text prompt matters more than anything else. "road network" and
    "highway" work well; the bare word "street" can return nothing at all. If a
    prompt scores badly, change the wording first.
  - Recall is usually high (~0.99) and precision low (~0.32): the model finds
    roads but paints them too wide. That is expected, not a failure. A higher
    threshold trades recall for precision.
  - sweep_threshold is far cheaper than re-segmenting. Use it to choose a
    threshold before trying another prompt.
  - upscale=2 helps because roads are thin at 1 m/pixel; it costs time.

When the user complains about one part of the map:
  - Use refine_area with the area they named (top-left, bottom-right, ...) or a
    bounding box if they drew one. Do NOT re-run the whole region.
  - Change something for the rework: the defaults already failed there, so try
    a different prompt or upscale=2.
  - Report what changed, with numbers.

Style:
  - Say what you are about to do in one short sentence before doing it.
  - Report real numbers from tool output. Never claim an improvement you did
    not measure.
  - When you are done, give a short plain-language summary: how much road you
    found, how confident it is, and anything you would still improve.
  - Be concise. No preamble, no restating the question.
"""


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class AgentEvent:
    """One observable step, streamed to the UI."""

    type: str      # status|thinking|tool_call|tool_result|message|done|error
    at: str = field(default_factory=_utc)
    step: int = 0
    text: str = ""
    tool: str = ""
    args: dict = field(default_factory=dict)
    result: Any = None
    duration_s: float = 0.0
    summary: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- #
# readable summaries of tool results, so the UI is not a wall of JSON
# --------------------------------------------------------------------------- #

def summarise(tool: str, result: Any) -> str:
    if not isinstance(result, dict):
        return ""
    if result.get("ok") is False:
        return f"failed: {str(result.get('error'))[:160]}"

    def pct(v):
        return f"{100 * float(v):.1f}%"

    try:
        if tool == "tile_region":
            return (f"{result['n_chips']} chips of {result['chip_size']}px "
                    f"({result['overlap']}px overlap)")
        if tool in ("segment_chips",):
            return (f"{result['n_chips']} chips, {result['total_instances']} "
                    f"instances, {pct(result['mean_coverage'])} covered, "
                    f"prompt '{result['prompt']}'")
        if tool == "stitch_result":
            return f"mask covers {pct(result['mask_fraction'])} of the region"
        if tool == "evaluate_result":
            return (f"IoU {result['iou']:.3f}, F1 {result['f1']:.3f} "
                    f"(P {result['precision']:.2f} / R {result['recall']:.2f}), "
                    f"relaxed F1 {result['relaxed_f1']:.3f}")
        if tool == "sweep_threshold":
            b = result.get("best_f1") or {}
            return (f"best F1 {b.get('f1', 0):.3f} at threshold "
                    f"{b.get('threshold')}")
        if tool == "vectorize_result":
            return (f"{result['n_features']} centrelines, "
                    f"{result['total_length_km']} km, mean confidence "
                    f"{pct(result.get('mean_confidence', 0))}")
        if tool == "refine_area":
            m = result.get("metrics") or {}
            base = f"reworked {result['n_chips']} chips in {result.get('note') or 'area'}"
            return base + (f" -> IoU {m['iou']:.3f}, F1 {m['f1']:.3f}" if m else "")
        if tool == "create_region_job":
            r = result.get("region", {})
            return f"job {result.get('job_id')}, {r.get('n_tiles')} tiles, {r.get('width')}x{r.get('height')}px"
        if tool == "list_available_tiles":
            return f"{result.get('total')} tiles in split"
        if tool == "low_confidence_roads":
            lo = result.get("lowest") or []
            return (f"{result.get('total')} roads; weakest at "
                    f"{lo[0]['confidence_pct']}%" if lo else "no roads yet")
        if tool == "get_job_status":
            return f"stage {result.get('stage')}"
    except Exception:
        pass
    return ""


# --------------------------------------------------------------------------- #
# conversation
# --------------------------------------------------------------------------- #

class Conversation:
    """Chat history for one job, persisted so a reload does not lose context."""

    def __init__(self, path: Path, system_prompt: str = SYSTEM_PROMPT) -> None:
        self.path = Path(path)
        self.system_prompt = system_prompt
        self.messages: list[dict] = []
        if self.path.exists():
            try:
                self.messages = json.loads(self.path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                self.messages = []

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self.messages, indent=1, default=str), encoding="utf-8"
        )

    def reset(self) -> None:
        self.messages = []
        self.save()

    def for_llm(self, context: str = "") -> list[dict]:
        system = self.system_prompt + (f"\n\nCurrent job context:\n{context}"
                                       if context else "")
        return [{"role": "system", "content": system}, *self._trimmed()]

    def _trimmed(self, keep: int = 60) -> list[dict]:
        """Keep the transcript bounded without orphaning tool replies."""
        if len(self.messages) <= keep:
            return self.messages
        cut = self.messages[-keep:]
        while cut and cut[0].get("role") == "tool":
            cut = cut[1:]
        return cut

    def visible(self) -> list[dict]:
        """Just the user/assistant turns, for rendering the chat panel."""
        out = []
        for m in self.messages:
            if m.get("role") == "user":
                out.append({"role": "user", "content": m.get("content", "")})
            elif m.get("role") == "assistant" and m.get("content"):
                out.append({"role": "assistant", "content": m["content"]})
        return out


# --------------------------------------------------------------------------- #
# the loop
# --------------------------------------------------------------------------- #

class RoadAgent:
    """LLM tool-calling loop over the shared MCP road tools."""

    def __init__(
        self,
        *,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        max_steps: int | None = None,
    ) -> None:
        s = get_settings()
        self.model = model or s.llm_model
        self.api_key = api_key or s.openai_api_key
        self.base_url = base_url or s.openai_base_url
        self.max_steps = max_steps or s.llm_max_steps

    def _client(self):
        from openai import AsyncOpenAI

        if not self.api_key:
            raise RuntimeError("no LLM API key configured; set OPENAI_API_KEY")
        return AsyncOpenAI(api_key=self.api_key, base_url=self.base_url,
                           timeout=180.0, max_retries=2)

    async def chat(
        self,
        conversation: Conversation,
        user_message: str,
        *,
        context: str = "",
    ) -> AsyncIterator[AgentEvent]:
        """Handle one user turn, calling tools until the agent replies."""
        client = self._client()

        try:
            listed = await mcp_client.list_tools()
        except Exception as exc:
            yield AgentEvent(type="error", text=f"MCP unavailable: {exc}")
            return

        tools = [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": (t.description or "")[:1024],
                    "parameters": json.loads(json.dumps(
                        t.input_schema or {"type": "object", "properties": {}}
                    )),
                },
            }
            for t in listed
        ]

        conversation.messages.append({"role": "user", "content": user_message})
        conversation.save()
        yield AgentEvent(type="status", text=f"{len(tools)} tools available",
                         result=[t["function"]["name"] for t in tools])

        for step in range(1, self.max_steps + 1):
            t0 = time.perf_counter()
            try:
                response = await client.chat.completions.create(
                    model=self.model,
                    messages=conversation.for_llm(context),
                    tools=tools,
                    tool_choice="auto",
                    temperature=0.2,
                    max_tokens=1600,
                )
            except Exception as exc:
                yield AgentEvent(type="error", step=step,
                                 text=f"LLM call failed: {exc}")
                return

            msg = response.choices[0].message
            calls = msg.tool_calls or []
            content = (msg.content or "").strip()

            entry: dict = {"role": "assistant", "content": content}
            if calls:
                entry["tool_calls"] = [
                    {"id": c.id, "type": "function",
                     "function": {"name": c.function.name,
                                  "arguments": c.function.arguments}}
                    for c in calls
                ]
            conversation.messages.append(entry)
            conversation.save()

            if content:
                yield AgentEvent(
                    type="thinking" if calls else "message",
                    step=step, text=content,
                    duration_s=time.perf_counter() - t0,
                )

            if not calls:
                yield AgentEvent(type="done", step=step, text=content)
                return

            for call in calls:
                name = call.function.name
                try:
                    args = json.loads(call.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}

                yield AgentEvent(type="tool_call", step=step, tool=name, args=args)

                ts = time.perf_counter()
                try:
                    parsed = await mcp_client.call(name, args)
                    text = (parsed if isinstance(parsed, str)
                            else json.dumps(parsed, default=str))
                except Exception as exc:
                    parsed = {"ok": False, "error": str(exc)}
                    text = json.dumps(parsed)

                yield AgentEvent(
                    type="tool_result", step=step, tool=name, args=args,
                    result=parsed, duration_s=time.perf_counter() - ts,
                    summary=summarise(name, parsed),
                )
                conversation.messages.append({
                    "role": "tool", "tool_call_id": call.id,
                    "content": text[:12000],
                })
            conversation.save()

        yield AgentEvent(type="error", step=self.max_steps,
                         text=f"stopped after {self.max_steps} steps")
