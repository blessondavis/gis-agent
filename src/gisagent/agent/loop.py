"""The agent: a conversational harness over the MCP road tools.

The person talks to it in plain language ("the top-left corner is missing
roads, redo it") and it decides which tools to call. Conversation state is
persisted per job, so a session survives a page reload.

Why an agent rather than a fixed script: the best threshold and settings vary
per region, and a person looking at the map is the fastest way to find the
parts that came out wrong. That is a search problem with a measurable
objective and a human in the loop.

The harness around the model borrows from xAI's Grok Build, adapted to a
domain where the artefacts are rasters and road networks rather than code:

* **A visible plan.** ``update_plan`` is a harness tool, not an MCP one: the
  model keeps a checklist the person can see, persisted with the job.
* **Plan mode.** Read-only tools only; the model investigates and proposes,
  and the person approves before anything expensive or destructive runs.
* **A stop gate.** The model may not finish while the output has changed since
  it last measured it, or while its summary quotes a number no tool returned.
  Grok Build gates ``/goal`` on an independent evidence review; here the
  evidence is the tool results themselves, so the check is deterministic.
* **Working memory, re-injected.** Before every model call the harness rebuilds
  a short ledger from the job on disk -- best score so far, settings already
  tried, the person's edits, the plan -- so trimming old turns never loses
  the facts that matter.
* **Spilling.** A large tool result is written to the job directory and the
  model sees a compact version plus the path, instead of blind truncation.

Events are typed (``plan``, ``tool_call``/``tool_result`` paired by call id,
``verify``, ...) so the UI can render each kind of thing as what it is.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import AsyncIterator
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from gisagent import mcp_client
from gisagent.config import get_settings

SYSTEM_PROMPT = """\
You are a GIS annotation agent. You extract road networks from satellite
imagery by calling tools, and you improve results by measuring them. A person
is looking at the map beside you and may be finishing the network by hand.

How to work
- For anything beyond a single question, first call update_plan with 3-6 short
  steps, and keep it current (in_progress / done) as you go. The person sees it.
- Say what you are about to do in one short sentence before doing it.
- Measure after you change anything: evaluate_result when ground truth exists,
  check_topology when it does not. The harness will not let you finish while
  the output has changed since your last measurement.
- Only quote numbers a tool returned in this conversation. The harness checks.

Pipeline
  tile_region -> segment_chips -> stitch_result -> vectorize_result -> measure
- segment_chips backend 'unet' (the default) was trained on 1 m/px aerial
  roads and is strong in suburbs and dense cities. 'sam3' is zero-shot and
  text-prompted; only for sam3 do the prompt wording and threshold matter.
- To choose a mask threshold or vectoriser settings, use try_candidates
  (isolated, ranked, takes seconds) and then apply_candidate the winner. That
  is cheaper and safer than re-running over the live result.
- Match try_candidates' objective to what the person wants: "precision" when
  a wrong road is worse than a missing one (base maps, navigation), "recall"
  when missing roads are worse (they will review and delete), else
  "balanced". Judge success by the same measure: correctness is the share of
  drawn roads that are real, completeness the share of real roads found.
- refine_area re-runs one area only. Use it when the person points at a bad
  area (they may attach a bbox_wgs84). Change something for the rework.

The person's edits
- They draw, reshape and delete roads in the map. Their edits are layered over
  your output and survive any re-run. Treat them as correct: never undo them
  and never re-add a road they deleted.
- network_status shows their edits and, with labels, how complete the network
  is. suggest_missing_roads finds candidate roads for them to review; tell them
  they can press R in the map to step through the suggestions.

Finishing
- Give a short plain-language summary: how much road, how good (measured
  numbers), and what is left for a person to do.
- Be concise. No preamble, no restating the question.
"""

PLAN_MODE = """\

PLAN MODE: you may only investigate. The tools that change the result are not
available. Look at the job, measure what exists, then call update_plan with a
concrete plan (which tools, which settings, how you will measure success) and
end with a short message asking the person to approve it.
"""

# Tools that cannot change the live result. Plan mode offers only these.
READ_ONLY = frozenset({
    "list_available_tiles", "list_jobs", "get_job_status", "inspect_chip",
    "evaluate_result", "sweep_threshold", "low_confidence_roads",
    "check_topology", "critique_annotation", "network_status",
    "suggest_missing_roads", "try_candidates", "qgis_version",
    "list_qgis_algorithms",
})
# Tools that change the live result, after which a measurement is owed.
MUTATING = frozenset({
    "segment_chips", "stitch_result", "vectorize_result", "refine_area",
    "repair_geometry", "apply_candidate", "run_qgis_algorithm",
})
# Tools that measure the live result.
VERIFYING = frozenset({
    "evaluate_result", "check_topology", "critique_annotation", "network_status",
})

MAX_GATES = 2            # stop-gate continuations per turn
SPILL_CHARS = 6000       # tool results larger than this go to disk
HARNESS_PREFIX = "[harness]"

PLAN_TOOL = {
    "type": "function",
    "function": {
        "name": "update_plan",
        "description": (
            "Show the person your plan as a checklist and keep it current. "
            "Call it at the start of multi-step work and whenever a step starts "
            "or finishes. Replaces the whole list each time."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "steps": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "status": {"type": "string", "enum": [
                                "pending", "in_progress", "done", "skipped"]},
                        },
                        "required": ["title", "status"],
                    },
                },
                "note": {"type": "string"},
            },
            "required": ["steps"],
        },
    },
}


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class AgentEvent:
    """One observable step, streamed to the UI."""

    type: str      # status|plan|thinking|tool_call|tool_result|verify|message|done|error|plan_ready
    at: str = field(default_factory=_utc)
    step: int = 0
    text: str = ""
    tool: str = ""
    call_id: str = ""
    args: dict = field(default_factory=dict)
    result: Any = None
    duration_s: float = 0.0
    summary: str = ""
    ok: bool = True

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
        if tool == "segment_chips":
            return (f"{result['n_chips']} chips, {pct(result['mean_coverage'])} "
                    f"covered")
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
                    f"{result['total_length_km']} km")
        if tool == "refine_area":
            m = result.get("metrics") or {}
            base = f"reworked {result['n_chips']} chips in {result.get('note') or 'area'}"
            return base + (f" -> IoU {m['iou']:.3f}, F1 {m['f1']:.3f}" if m else "")
        if tool == "try_candidates":
            best = result["candidates"][0]
            return (f"{len(result['candidates'])} candidates for {result['objective']}; "
                    f"best {best['id']} {best['params']} score {best['score']:.3f}")
        if tool == "apply_candidate":
            m = result.get("metrics") or {}
            return f"applied {result['applied']}" + (f" -> IoU {m['iou']:.3f}" if m else "")
        if tool == "check_topology":
            return (f"score {result['score']:.2f} ({result['verdict']}), "
                    f"{result['n_components']} pieces, {result['n_dangles']} dead ends")
        if tool == "network_status":
            s = result.get("stats", {})
            out = (f"{s.get('total_length_km')} km, {s.get('n_human', 0)} drawn by "
                   f"the person, {s.get('n_ops', 0)} edits")
            sc = (result.get("score") or {}).get("overall")
            return out + (f", completeness {pct(sc['completeness'])}" if sc else "")
        if tool == "suggest_missing_roads":
            return f"{result.get('gap', 0)} gaps, {result.get('missed', 0)} possible roads"
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
# working memory: facts rebuilt from disk before every model call
# --------------------------------------------------------------------------- #

def _fmt(v, d=3):
    return f"{v:.{d}f}" if isinstance(v, (int, float)) else str(v)


def working_memory(job) -> str:
    """A short ledger of what the job knows, derived from its files."""
    m = job.manifest
    region = m.get("region") or {}
    lines = [
        f"job {job.job_id} | {region.get('width')}x{region.get('height')} px, "
        f"{region.get('crs')} | ground truth: {'yes' if job.has_truth() else 'no'} "
        f"| default backend: {get_settings().segment_backend}",
        f"stage: {m.get('stage', 'created')}",
    ]

    # settings tried, paired with the score they produced
    tried, best, last_thr = [], None, None
    for s in m.get("stages", []):
        p, r = s.get("params") or {}, s.get("result") or {}
        if s.get("stage") == "segmented" and not p.get("refine"):
            tried.append(f"segment prompt={p.get('prompt')!r} upscale={p.get('upscale')}")
        elif s.get("stage") == "stitched":
            last_thr = p.get("threshold")
        elif s.get("stage") == "evaluated" and "iou" in r:
            tried.append(f"mask threshold {last_thr} -> IoU {_fmt(r['iou'])}, "
                         f"relaxed F1 {_fmt(r.get('relaxed_f1'))}")
            if best is None or r["iou"] > best[0]:
                best = (r["iou"], last_thr)
    if tried:
        lines.append("tried: " + "; ".join(dict.fromkeys(tried[-8:])))
    if best:
        lines.append(f"best IoU so far: {_fmt(best[0])} (mask threshold {best[1]})")
    if m.get("metrics"):
        x = m["metrics"]
        lines.append(f"latest metrics: IoU {_fmt(x['iou'])}, F1 {_fmt(x['f1'])}, "
                     f"relaxed F1 {_fmt(x.get('relaxed_f1'))}")
    refs = m.get("refinements") or []
    if refs:
        lines.append("areas reworked: " + ", ".join(r.get("note") or "?" for r in refs[-4:]))
    topo = m.get("topology")
    if topo:
        lines.append(f"last topology check: score {_fmt(topo.get('score'), 2)} "
                     f"({topo.get('verdict')})")
    cand = job.dir / "candidates" / "index.json"
    if cand.exists():
        try:
            c = json.loads(cand.read_text(encoding="utf-8"))
            b = c["candidates"][0]
            lines.append(f"last candidates ({c.get('objective', 'balanced')}): best "
                         f"{b['id']} {b['params']} score {_fmt(b['score'])} "
                         f"of {len(c['candidates'])}")
        except (json.JSONDecodeError, KeyError, IndexError):
            pass
    if job.network_path.exists() or job.edits_path.exists():
        try:
            s = job.network().get("stats", {})
            lines.append(
                f"network: {s.get('total_length_km')} km; the person drew "
                f"{s.get('n_human', 0)} roads ({_fmt(s.get('human_length_m', 0) / 1000, 2)} km) "
                f"and removed {s.get('n_deleted', 0) + s.get('n_superseded', 0)} "
                f"({s.get('n_ops', 0)} edits)")
        except Exception:
            pass
    plan = load_plan(job.dir)
    if plan:
        steps = " | ".join(f"[{x['status']}] {x['title']}" for x in plan["steps"])
        lines.append(f"plan: {steps}")
    return "\n".join(lines)


def load_plan(job_dir: Path) -> dict | None:
    p = Path(job_dir) / "plan.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _save_plan(job_dir: Path, args: dict) -> dict:
    steps = []
    for s in (args.get("steps") or [])[:12]:
        title = str(s.get("title", "")).strip()[:120]
        status = s.get("status") if s.get("status") in (
            "pending", "in_progress", "done", "skipped") else "pending"
        if title:
            steps.append({"title": title, "status": status})
    if not steps:
        raise ValueError("a plan needs at least one step with a title")
    plan = {"steps": steps, "note": str(args.get("note", ""))[:300], "updated_at": _utc()}
    (Path(job_dir) / "plan.json").write_text(json.dumps(plan, indent=1), encoding="utf-8")
    return plan


# --------------------------------------------------------------------------- #
# spilling large results
# --------------------------------------------------------------------------- #

def compact(obj: Any, depth: int = 0) -> Any:
    """Keep the shape and the scalars; cut long lists and deep nesting."""
    if isinstance(obj, dict):
        if depth >= 3:
            return f"{{...{len(obj)} keys}}"
        return {k: compact(v, depth + 1) for k, v in obj.items()}
    if isinstance(obj, list):
        if all(not isinstance(v, (dict, list)) for v in obj):
            return obj if len(obj) <= 12 else obj[:12] + [f"...{len(obj) - 12} more"]
        head = [compact(v, depth + 1) for v in obj[:5]]
        return head + ([f"...{len(obj) - 5} more"] if len(obj) > 5 else [])
    if isinstance(obj, str) and len(obj) > 400:
        return obj[:400] + "..."
    return obj


def for_model(parsed: Any, spill_dir: Path | None, stem: str) -> tuple[str, str]:
    """(text for the model, path it was spilled to or "")."""
    text = parsed if isinstance(parsed, str) else json.dumps(parsed, default=str)
    if len(text) <= SPILL_CHARS or spill_dir is None:
        return text[:SPILL_CHARS], ""
    spill_dir.mkdir(parents=True, exist_ok=True)
    path = spill_dir / f"{stem}.json"
    path.write_text(text, encoding="utf-8")
    small = compact(parsed) if not isinstance(parsed, str) else parsed[:SPILL_CHARS]
    if isinstance(small, dict):
        small["_note"] = f"large result: compacted here, full copy at {path}"
    out = json.dumps(small, default=str)
    return out[:SPILL_CHARS], str(path)


# --------------------------------------------------------------------------- #
# the evidence check
# --------------------------------------------------------------------------- #

_METRIC = re.compile(r"(?<![\d.])0\.\d{2,4}(?!\d)")
_ANY_NUMBER = re.compile(r"-?\d+\.\d+")


def unsupported_numbers(text: str, evidence: list[str]) -> list[str]:
    """Metric-like numbers (0.xx) in ``text`` that no tool result contains.

    Matching is at the precision the model wrote: "0.79" is supported by a
    tool's 0.7914. Deliberately narrow -- scores are what people act on, and
    what a model is most tempted to round up or remember wrong.
    """
    have2, have3, have4 = set(), set(), set()
    for blob in evidence:
        for s in _ANY_NUMBER.findall(blob):
            v = float(s)
            have2.add(round(v, 2))
            have3.add(round(v, 3))
            have4.add(round(v, 4))
    bad = []
    for s in dict.fromkeys(_METRIC.findall(text)):
        digits = len(s.split(".")[1])
        v = float(s)
        pool = {2: have2, 3: have3, 4: have4}[digits]
        if round(v, digits) not in pool:
            bad.append(s)
    return bad


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

    def truncate(self, n: int) -> None:
        self.messages = self.messages[:max(0, n)]
        self.save()

    def for_llm(self, context: str = "", extra_system: str = "") -> list[dict]:
        system = self.system_prompt + extra_system + (
            f"\n\nWorking memory (rebuilt from the job before every step):\n{context}"
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
        """The person's and the agent's turns, for the chat panel.

        Each entry carries its index into ``messages``, which is what a
        rewind truncates to.
        """
        out = []
        for i, m in enumerate(self.messages):
            content = m.get("content") or ""
            if m.get("role") == "user" and not content.startswith(HARNESS_PREFIX):
                out.append({"role": "user", "content": content, "index": i})
            elif m.get("role") == "assistant" and content:
                out.append({"role": "assistant", "content": content, "index": i})
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
        client=None,
        call_tool=None,
        list_tools=None,
    ) -> None:
        s = get_settings()
        self.model = model or s.llm_model
        self.api_key = api_key or s.openai_api_key
        self.base_url = base_url or s.openai_base_url
        self.max_steps = max_steps or s.llm_max_steps
        # injectable, so the harness logic can be tested without a model or GPU
        self._client_override = client
        self._call_tool = call_tool or mcp_client.call
        self._list_tools = list_tools or mcp_client.list_tools

    def _client(self):
        if self._client_override is not None:
            return self._client_override
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
        job_id: str | None = None,
        context: str = "",
        mode: str = "work",
    ) -> AsyncIterator[AgentEvent]:
        """Handle one user turn, calling tools until the agent replies.

        ``mode`` is "work" (all tools) or "plan" (read-only tools; the model
        proposes a plan and stops).
        """
        from gisagent import pipeline

        client = self._client()
        job = None
        if job_id:
            try:
                job = pipeline.get_job(job_id)
            except FileNotFoundError:
                job = None

        try:
            listed = await self._list_tools()
        except Exception as exc:
            yield AgentEvent(type="error", text=f"MCP unavailable: {exc}", ok=False)
            return

        tools = [PLAN_TOOL] + [
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
            if mode != "plan" or t.name in READ_ONLY
        ]

        conversation.messages.append({"role": "user", "content": user_message})
        conversation.save()
        yield AgentEvent(type="status", text=f"{len(tools)} tools available ({mode} mode)",
                         result=[t["function"]["name"] for t in tools])

        turn_start = len(conversation.messages)
        dirty = False            # the live result changed since the last measurement
        gates = 0
        spill_dir = job.dir / "spill" if job else None

        for step in range(1, self.max_steps + 1):
            if job is not None:
                job = pipeline.get_job(job.job_id)     # the MCP process changed it
                context = working_memory(job)
            t0 = time.perf_counter()
            try:
                response = await client.chat.completions.create(
                    model=self.model,
                    messages=conversation.for_llm(
                        context, PLAN_MODE if mode == "plan" else ""),
                    tools=tools,
                    tool_choice="auto",
                    temperature=0.2,
                    max_tokens=1600,
                )
            except Exception as exc:
                yield AgentEvent(type="error", step=step, ok=False,
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
                # ---- the stop gate: is this answer supported? -------------- #
                problems = []
                if mode != "plan" and dirty:
                    problems.append(
                        "the result changed after your last measurement. Measure it "
                        "(evaluate_result with ground truth, check_topology without) "
                        "and report what you measured.")
                # evidence: what tools returned, the ledger, and the settings the
                # agent itself chose (quoting its own threshold is not a claim)
                evidence = [m.get("content") or "" for m in conversation.messages
                            if m.get("role") == "tool"] + [context]
                evidence += [c["function"]["arguments"] for m in conversation.messages
                             for c in m.get("tool_calls") or []]
                bad = unsupported_numbers(content, evidence)
                if bad:
                    problems.append(
                        f"your reply quotes {', '.join(bad)}, which no tool returned. "
                        "Check with a tool or correct the numbers.")
                if problems and gates < MAX_GATES:
                    gates += 1
                    note = f"{HARNESS_PREFIX} Before you finish: " + " Also, ".join(problems)
                    conversation.messages.append({"role": "user", "content": note})
                    conversation.save()
                    yield AgentEvent(type="verify", step=step, ok=False,
                                     text=" ".join(problems))
                    continue
                if problems:
                    yield AgentEvent(type="verify", step=step, ok=False,
                                     text="finished with unresolved checks: " + " ".join(problems))
                elif len(conversation.messages) > turn_start + 1:
                    yield AgentEvent(type="verify", step=step, ok=True,
                                     text="output measured after the last change; "
                                          "quoted scores match tool results")
                if mode == "plan":
                    yield AgentEvent(type="plan_ready", step=step, text=content)
                yield AgentEvent(type="done", step=step, text=content)
                return

            for call in calls:
                name = call.function.name
                try:
                    args = json.loads(call.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}

                yield AgentEvent(type="tool_call", step=step, tool=name,
                                 call_id=call.id, args=args)
                ts = time.perf_counter()

                if name == "update_plan":
                    try:
                        if job is None:
                            raise ValueError("no job to attach a plan to")
                        plan = _save_plan(job.dir, args)
                        parsed = {"ok": True, "steps": len(plan["steps"])}
                        yield AgentEvent(type="plan", step=step, call_id=call.id,
                                         result=plan)
                    except ValueError as exc:
                        parsed = {"ok": False, "error": str(exc)}
                elif mode == "plan" and name not in READ_ONLY:
                    parsed = {"ok": False, "error": "plan mode: this tool changes the "
                              "result; propose it in the plan instead"}
                else:
                    try:
                        parsed = await self._call_tool(name, args)
                    except Exception as exc:
                        parsed = {"ok": False, "error": str(exc)}

                ok = not (isinstance(parsed, dict) and parsed.get("ok") is False)
                if ok and name in MUTATING:
                    # tools that measure their own output (refine_area and
                    # apply_candidate with labels) settle the debt themselves
                    dirty = not (isinstance(parsed, dict) and parsed.get("metrics"))
                elif ok and name in VERIFYING:
                    dirty = False

                text, spilled = for_model(parsed, spill_dir, f"{step:03d}-{name}-{call.id[-6:]}")
                yield AgentEvent(
                    type="tool_result", step=step, tool=name, call_id=call.id,
                    args=args, result=parsed, ok=ok,
                    duration_s=time.perf_counter() - ts,
                    summary=summarise(name, parsed)
                    + (" (large result spilled to disk)" if spilled else ""),
                )
                conversation.messages.append({
                    "role": "tool", "tool_call_id": call.id, "content": text,
                })
            conversation.save()

        yield AgentEvent(type="error", step=self.max_steps, ok=False,
                         text=f"stopped after {self.max_steps} steps")
