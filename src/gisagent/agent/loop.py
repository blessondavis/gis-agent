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

import asyncio
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
- The harness measures every change to the network itself -- against ground
  truth when there is some, else by expected precision/recall under the
  model's confidence -- and appends the score to the tool result. Use that
  score to decide what to keep. It also keeps the best result: if a change made
  things worse, call restore_best.
- When the person says what matters ("wrong roads are worse", "don't miss
  any"), call set_objective so the harness scores by that.
- Only quote numbers a tool or the harness returned. The harness checks.
- A call refused with "[harness rule]" tells you why and what to do instead.
  Do that; do not retry the same call.

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
- refine_area re-runs the model on one area (the person may attach a
  bbox_wgs84). With the U-Net it only changes anything with backend='sam3':
  the U-Net is deterministic at native scale and upscaling it was measured to
  make results worse. For most fixes, try_candidates -- or the person's own
  edits via suggest_missing_roads -- are the better lever.

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
MAX_GATES_TASK = 8       # an autonomous task is sent back until it is done, within reason
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


RESTORE_TOOL = {
    "type": "function",
    "function": {
        "name": "restore_best",
        "description": (
            "Put back the best-scoring result the harness has measured in this task "
            "(mask, centrelines and confidence map). Use it when a change made the "
            "score worse."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
}

OBJECTIVE_TOOL = {
    "type": "function",
    "function": {
        "name": "set_objective",
        "description": (
            "Change what 'better' means for this job, when the person states a "
            "preference: 'precision' if a wrong road is worse than a missing one, "
            "'recall' if a missing road is worse, else 'balanced'. The harness "
            "re-scores everything measured so far under the new objective."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "objective": {"type": "string", "enum": ["balanced", "precision", "recall"]},
                "reason": {"type": "string"},
            },
            "required": ["objective"],
        },
    },
}

HARNESS_TOOLS = {"update_plan", "restore_best", "set_objective"}


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
        if tool == "restore_best":
            return f"restored the best result ({result.get('score', 0):.3f})"
        if tool == "set_objective":
            return f"objective is now {result.get('objective')}"
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
        spec=None,
    ) -> AsyncIterator[AgentEvent]:
        """Handle one user turn, calling tools until the agent replies.

        ``mode``:

        * "work" -- a conversation turn. All tools; the harness measures every
          change and keeps the best, but the agent decides when it is done.
        * "plan" -- read-only tools; the model proposes a plan and stops.
        * "task" -- the agent annotates the region on its own. The rules in
          :mod:`gisagent.agent.rules` are enforced: order, budgets, no
          repeats, plateau, and a definition of done it cannot finish
          without. Ends with a report whose verdict comes from the checks.

        ``spec`` (a :class:`~gisagent.agent.rules.TaskSpec`) sets the task's
        objective, targets and budgets.
        """
        from gisagent import pipeline
        from gisagent.agent import rules

        client = self._client()
        job = None
        if job_id:
            try:
                job = pipeline.get_job(job_id)
            except FileNotFoundError:
                job = None
        task = mode == "task"
        t_start = time.perf_counter()

        # The ledger persists with the job: a task starts a fresh one; a
        # conversation turn continues whatever is there (so keep-best and the
        # score history survive between turns).
        ledger = None
        if job is not None:
            ledger = None if task else rules.Ledger.load(job)
            if ledger is None:
                ledger = rules.Ledger(spec=spec or rules.TaskSpec())
                ledger.save(job)
            elif spec is not None:
                ledger.spec = spec
                ledger.save(job)
            ledger.turn_start = len(ledger.calls)

        try:
            listed = await self._list_tools()
        except Exception as exc:
            yield AgentEvent(type="error", text=f"MCP unavailable: {exc}", ok=False)
            return

        # A conversation is bound to one job. The harness fills in job_id on
        # every call rather than trusting the model to copy a 22-character id:
        # a live run mistyped it, and the failure cost a detour.
        takes_job = {t.name for t in listed
                     if "job_id" in ((t.input_schema or {}).get("properties") or {})}

        extra = [RESTORE_TOOL, OBJECTIVE_TOOL] if mode != "plan" and job is not None else []
        tools = [PLAN_TOOL] + extra + [
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
        calls_since_gate = 0     # tool calls since the harness last sent the agent back
        idle_gates = 0           # consecutive send-backs answered with no action
        max_gates = MAX_GATES_TASK if task else MAX_GATES
        spill_dir = job.dir / "spill" if job else None
        extra_system = PLAN_MODE if mode == "plan" else ""
        if task and job is not None:
            extra_system = rules.rules_text(ledger.spec, labelled=job.has_truth())

        def finish_report():
            if task and job is not None:
                rep = rules.write_report(job, ledger, duration_s=time.perf_counter() - t_start)
                return AgentEvent(type="report", result=rep, ok=rep["status"] in (
                    "complete", "best_effort"), text=f"{rep['status']}: {rep['reason']}")
            return None

        for step in range(1, self.max_steps + 1):
            if job is not None:
                job = pipeline.get_job(job.job_id)     # the MCP process changed it
                context = working_memory(job) + ledger_memory(job, ledger, task=task)
            t0 = time.perf_counter()
            try:
                response = await client.chat.completions.create(
                    model=self.model,
                    messages=conversation.for_llm(context, extra_system),
                    tools=tools,
                    tool_choice="auto",
                    temperature=0.2,
                    max_tokens=1600,
                )
            except Exception as exc:
                yield AgentEvent(type="error", step=step, ok=False,
                                 text=f"LLM call failed: {exc}")
                rep = finish_report()
                if rep:
                    yield rep
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
                if task and job is not None:
                    # the definition of done, decided by code
                    job = pipeline.get_job(job.job_id)
                    unmet = await asyncio.to_thread(rules.unmet, job, ledger)
                    idle_gates = idle_gates + 1 if (unmet and gates and calls_since_gate == 0) else 0
                    if idle_gates >= 2:
                        # Sent back twice and did nothing either time: another
                        # identical demand would just burn steps. Stop, and let
                        # the report say what was left undone.
                        yield AgentEvent(type="verify", step=step, ok=False,
                                         text="stopped: the agent made no further progress "
                                              "after being sent back. Unmet: " + " | ".join(unmet))
                        rep = finish_report()
                        if rep:
                            yield rep
                        yield AgentEvent(type="done", step=step, text=content)
                        return
                    problems += unmet
                elif ledger is not None and ledger.worse_than_best():
                    # even in conversation: never leave the person a worse result
                    # than one already measured
                    live, ref = ledger.last_compare
                    problems.append(
                        f"the live result scores {live:.3f}, below the best measured "
                        f"{ref:.3f} (after {ledger.best.after}), compared like for like. "
                        "restore_best, or tell the person why not.")
                if problems and gates < max_gates:
                    gates += 1
                    calls_since_gate = 0
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
                rep = finish_report()
                if rep:
                    yield rep
                yield AgentEvent(type="done", step=step, text=content)
                return

            for call in calls:
                name = call.function.name
                calls_since_gate += 1
                try:
                    args = json.loads(call.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                if not isinstance(args, dict):
                    args = {}
                if job is not None and name in takes_job:
                    args["job_id"] = job.job_id

                yield AgentEvent(type="tool_call", step=step, tool=name,
                                 call_id=call.id, args=args)
                ts = time.perf_counter()

                if job is not None:
                    job = pipeline.get_job(job.job_id)
                state_before = rules.state_signature(job)
                net_before = _net_signature(job)
                refused = False
                measured = None

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
                    refused = True
                elif name == "set_objective" and ledger is not None:
                    parsed = _set_objective(job, ledger, args)
                elif name == "restore_best" and ledger is not None:
                    try:
                        parsed = {"ok": True, **await asyncio.to_thread(
                            rules.restore_best, job, ledger)}
                    except RuntimeError as exc:
                        parsed = {"ok": False, "error": str(exc)}
                else:
                    reason = rules.precheck(name, args, job, ledger, task=task)
                    if reason:
                        parsed = {"ok": False, "error": f"[harness rule] {reason}"}
                        refused = True
                    else:
                        if name == "try_candidates" and ledger is not None:
                            # the task's objective, not whatever the model typed
                            if args.get("objective") not in (None, ledger.spec.objective):
                                args["objective_requested"] = args["objective"]
                            args["objective"] = ledger.spec.objective
                        call_args = {k: v for k, v in args.items() if k != "objective_requested"}
                        try:
                            parsed = await self._call_tool(name, call_args)
                        except Exception as exc:
                            parsed = {"ok": False, "error": str(exc)}

                ok = not (isinstance(parsed, dict) and parsed.get("ok") is False)
                if name not in ("update_plan",):
                    rules.note_call(ledger, name, args, ok, refused=refused, state=state_before)
                if ok and name in MUTATING | {"restore_best"}:
                    # tools that measure their own output (refine_area and
                    # apply_candidate with labels) settle the debt themselves
                    dirty = not (isinstance(parsed, dict) and parsed.get("metrics"))
                elif ok and name in VERIFYING:
                    dirty = False

                if ok and job is not None and ledger is not None:
                    job = pipeline.get_job(job.job_id)
                    if name == "try_candidates":
                        ledger.candidate_versions.append(rules.confidence_version(job))
                    if name == "suggest_missing_roads":
                        ledger.suggested_after = len(ledger.measurements) - 1
                    # the harness measures every change to the network itself
                    if name in rules.CHANGES_RESULT and _net_signature(job) != net_before \
                            or name in ("apply_candidate", "restore_best"):
                        measured = await asyncio.to_thread(
                            rules.record, job, ledger, _describe(name, args))
                        if measured is not None:
                            dirty = False
                    ledger.save(job)
                if refused:
                    yield AgentEvent(type="rule", step=step, tool=name, call_id=call.id,
                                     ok=False, text=parsed["error"])

                text, spilled = for_model(parsed, spill_dir, f"{step:03d}-{name}-{call.id[-6:]}")
                if measured is not None:
                    note = _measure_note(measured, ledger)
                    text = f"{text}\n{HARNESS_PREFIX} {note}"
                    yield AgentEvent(type="measure", step=step, call_id=call.id,
                                     result=_measure_dict(measured, ledger), text=note)
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
        rep = finish_report()
        if rep:
            yield rep


# --------------------------------------------------------------------------- #
# harness helpers for the rules
# --------------------------------------------------------------------------- #

def _net_signature(job) -> str:
    """Changes when the delivered network could have changed."""
    if job is None:
        return ""
    parts = []
    for p in (job.vector_path, job.edits_path):
        try:
            st = p.stat()
            parts.append(f"{st.st_size}.{st.st_mtime_ns}")
        except FileNotFoundError:
            parts.append("-")
    return "|".join(parts)


def _describe(tool: str, args: dict) -> str:
    """A short label for the ledger: which call produced this state."""
    keep = {k: v for k, v in args.items()
            if k not in ("job_id", "note", "objective_requested") and v not in (None, "", [])}
    if tool == "try_candidates":
        keep.pop("variants", None)
    s = ", ".join(f"{k}={v}" for k, v in keep.items())
    return f"{tool}({s})"[:120]


def _measure_dict(m, ledger) -> dict:
    return {"score": m.score, "precision": m.precision, "recall": m.recall,
            "basis": m.basis, "objective": ledger.spec.objective, "best": m.best,
            "best_score": ledger.best.score if ledger.best else None,
            "km": m.km, "after": m.after}


def _measure_note(m, ledger) -> str:
    what = ("correctness / completeness vs ground truth" if m.basis == "ground truth"
            else "expected precision / recall under the model's confidence")
    best = ledger.best
    if m.best:
        verdict = "new best"
    else:
        live, ref = ledger.last_compare or (m.score, best.score)
        how = ("" if ledger.comparable(m, best) else
               " (the model was re-run, so both were judged under the best result's "
               "confidence map)")
        verdict = (f"not better than the best: {live:.3f} vs {ref:.3f}{how}, best after "
                   f"{best.after}" + ("; restore_best if you cannot beat it"
                                      if ledger.worse_than_best() else ""))
    return (f"measured: {ledger.spec.objective} score {m.score:.3f} "
            f"(precision {m.precision:.3f}, recall {m.recall:.3f}; {what}), "
            f"{m.km} km. {verdict}.")


def _set_objective(job, ledger, args: dict) -> dict:
    from gisagent.agent.rules import OBJECTIVES, TaskSpec
    from gisagent.evaluate.network import f_beta

    obj = args.get("objective")
    if obj not in OBJECTIVES:
        return {"ok": False, "error": f"objective must be one of {OBJECTIVES}"}
    ledger.spec = TaskSpec(**{**ledger.spec.to_dict(), "objective": obj})
    # re-score history under the new objective; precision/recall are stored
    for m in ledger.measurements:
        m.score = round(f_beta(m.precision, m.recall, ledger.spec.beta), 4)
        m.best = False
    if ledger.measurements:
        cur = ledger.measurements[-1]
        comparable = [i for i, m in enumerate(ledger.measurements) if ledger.comparable(m, cur)]
        ledger.best_index = max(comparable, key=lambda i: ledger.measurements[i].score)
        ledger.measurements[ledger.best_index].best = True
    ledger.save(job)
    return {"ok": True, "objective": obj, "reason": args.get("reason", ""),
            "best_score": ledger.best.score if ledger.best else None}


def ledger_memory(job, ledger, *, task: bool) -> str:
    """The rules' state, for the working memory the model reads each step."""
    if ledger is None:
        return ""
    from gisagent.agent import rules

    s = ledger.spec
    lines = [f"objective: {s.objective}"]
    cur, best = ledger.latest, ledger.best
    if cur:
        lines.append(f"harness score now {cur.score:.3f} ({cur.basis}); best {best.score:.3f} "
                     f"after {best.after}" if best else f"harness score now {cur.score:.3f}")
        if ledger.worse_than_best():
            lines.append("the live result is WORSE than the best: restore_best")
    if task:
        lines.append("budgets used: " + ", ".join(
            f"{t} {ledger.count(t)}/{getattr(s, k)}" for t, k in rules.BUDGETED.items()))
        if ledger.plateaued():
            lines.append("PLATEAU reached: stop improving, hand off and finish")
        left = rules.unmet(job, ledger)
        if left:
            lines.append("not done yet: " + " | ".join(left))
        else:
            lines.append("definition of done: all met -- finish with your report")
    return "\n" + "\n".join(lines)
