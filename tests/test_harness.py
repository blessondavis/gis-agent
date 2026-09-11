"""The agent harness, driven by a scripted fake model and fake tools.

No LLM, no MCP server, no GPU: the point is the harness's own behaviour --
the stop gate, the evidence check, the plan tool, plan mode and spilling.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from gisagent import pipeline
from gisagent.agent.loop import (
    SPILL_CHARS, Conversation, RoadAgent, compact, unsupported_numbers,
)


# --------------------------------------------------------------------------- #
# fakes
# --------------------------------------------------------------------------- #

def _call(name, args=None, cid=None):
    return SimpleNamespace(id=cid or f"call_{name}",
                           function=SimpleNamespace(name=name,
                                                    arguments=json.dumps(args or {})))


def _reply(content="", calls=None):
    msg = SimpleNamespace(content=content, tool_calls=calls or None)
    return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


class ScriptedModel:
    """Returns the scripted replies in order; records what it was sent."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.seen = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, **kw):
        self.seen.append(kw)
        return self.replies.pop(0)


TOOLS = [SimpleNamespace(name=n, description=n, input_schema={"type": "object", "properties": {}})
         for n in ("vectorize_result", "evaluate_result", "check_topology",
                   "get_job_status", "segment_chips")]


def fake_tools(results):
    calls = []

    async def call(name, args):
        calls.append(name)
        return results.get(name, {"ok": True})

    async def listing():
        return TOOLS

    return call, listing, calls


@pytest.fixture
def job(tmp_path, monkeypatch):
    """A job that has been through the pipeline, as far as the harness's
    precondition checks can tell (they only look at which files exist)."""
    monkeypatch.setattr(pipeline, "jobs_root", lambda: tmp_path)
    j = pipeline.new_job("harness-test")
    (j.chips_dir).mkdir()
    (j.chips_dir / "chips.json").write_text("[]")
    j.conf_dir.mkdir()
    (j.conf_dir / "c.npy").write_bytes(b"")
    for name in ("mask.tif", "confidence.tif", "truth.tif"):
        (j.dir / name).write_bytes(b"")
    (j.dir / "roads.geojson").write_text('{"type": "FeatureCollection", "features": []}')
    return j


async def run(agent, convo, message, **kw):
    return [ev async for ev in agent.chat(convo, message, **kw)]


# --------------------------------------------------------------------------- #
# the stop gate
# --------------------------------------------------------------------------- #

async def test_cannot_finish_after_a_change_without_measuring(job):
    model = ScriptedModel([
        _reply(calls=[_call("vectorize_result")]),
        _reply("Done, the roads look great."),                    # gated
        _reply(calls=[_call("evaluate_result")]),
        _reply("Done: IoU 0.58."),
    ])
    call, listing, calls = fake_tools({"evaluate_result": {"ok": True, "iou": 0.5812}})
    agent = RoadAgent(client=model, call_tool=call, list_tools=listing, max_steps=10)
    convo = Conversation(job.dir / "conversation.json")

    events = await run(agent, convo, "vectorize it", job_id=job.job_id)

    verifies = [e for e in events if e.type == "verify"]
    assert [v.ok for v in verifies] == [False, True]
    assert "changed after your last measurement" in verifies[0].text
    assert calls == ["vectorize_result", "evaluate_result"]
    assert events[-1].type == "done" and "0.58" in events[-1].text
    # the harness nudge is in the transcript but not in the visible chat
    assert any(m["content"].startswith("[harness]") for m in convo.messages
               if m["role"] == "user")
    assert [m["content"] for m in convo.visible() if m["role"] == "user"] == ["vectorize it"]


async def test_a_tool_that_measures_itself_settles_the_debt(job):
    model = ScriptedModel([
        _reply(calls=[_call("vectorize_result")]),
        _reply(calls=[_call("check_topology")]),
        _reply("Network is coherent."),
    ])
    call, listing, _ = fake_tools({})
    agent = RoadAgent(client=model, call_tool=call, list_tools=listing)
    events = await run(agent, Conversation(job.dir / "c.json"), "go", job_id=job.job_id)
    assert [e.ok for e in events if e.type == "verify"] == [True]


async def test_quoting_a_number_no_tool_returned_is_challenged(job):
    model = ScriptedModel([
        _reply(calls=[_call("evaluate_result")]),
        _reply("IoU is 0.91, excellent."),                        # made up
        _reply("Correction: IoU is 0.581."),
    ])
    call, listing, _ = fake_tools({"evaluate_result": {"ok": True, "iou": 0.5812}})
    agent = RoadAgent(client=model, call_tool=call, list_tools=listing)
    events = await run(agent, Conversation(job.dir / "c.json"), "score?", job_id=job.job_id)
    verifies = [e for e in events if e.type == "verify"]
    assert not verifies[0].ok and "0.91" in verifies[0].text
    assert verifies[-1].ok


async def test_the_gate_gives_up_after_its_budget(job):
    model = ScriptedModel([_reply(calls=[_call("vectorize_result")])]
                          + [_reply("done, trust me")] * 3)
    call, listing, _ = fake_tools({})
    agent = RoadAgent(client=model, call_tool=call, list_tools=listing)
    events = await run(agent, Conversation(job.dir / "c.json"), "go", job_id=job.job_id)
    assert events[-1].type == "done"
    assert "unresolved" in [e for e in events if e.type == "verify"][-1].text


def test_evidence_matching_is_at_the_precision_written():
    ev = ['{"iou": 0.5812, "relaxed_f1": 0.7914}']
    assert unsupported_numbers("IoU 0.58, relaxed F1 0.791", ev) == []
    assert unsupported_numbers("IoU 0.59", ev) == ["0.59"]
    assert unsupported_numbers("threshold 0.5 and 12 chips", ev) == []   # not metric-shaped


# --------------------------------------------------------------------------- #
# plan tool and plan mode
# --------------------------------------------------------------------------- #

async def test_update_plan_is_saved_and_streamed(job):
    steps = [{"title": "Segment", "status": "done"}, {"title": "Measure", "status": "in_progress"}]
    model = ScriptedModel([
        _reply(calls=[_call("update_plan", {"steps": steps})]),
        _reply("Planned."),
    ])
    call, listing, calls = fake_tools({})
    agent = RoadAgent(client=model, call_tool=call, list_tools=listing)
    events = await run(agent, Conversation(job.dir / "c.json"), "plan it", job_id=job.job_id)

    plan_ev = next(e for e in events if e.type == "plan")
    assert [s["title"] for s in plan_ev.result["steps"]] == ["Segment", "Measure"]
    assert json.loads((job.dir / "plan.json").read_text())["steps"][1]["status"] == "in_progress"
    assert calls == []                                  # handled by the harness, not MCP
    # and the plan is re-injected into the next model call's working memory
    assert "[in_progress] Measure" in model.seen[-1]["messages"][0]["content"]


async def test_plan_mode_offers_only_read_only_tools_and_refuses_the_rest(job):
    model = ScriptedModel([
        _reply(calls=[_call("segment_chips")]),
        _reply("Proposed plan: segment, then measure. Approve?"),
    ])
    call, listing, calls = fake_tools({})
    agent = RoadAgent(client=model, call_tool=call, list_tools=listing)
    events = await run(agent, Conversation(job.dir / "c.json"), "improve it",
                       job_id=job.job_id, mode="plan")

    offered = {t["function"]["name"] for t in model.seen[0]["tools"]}
    assert "segment_chips" not in offered and "vectorize_result" not in offered
    assert {"update_plan", "evaluate_result", "get_job_status"} <= offered
    assert calls == []                                  # the mutating call never ran
    assert "plan mode" in next(e for e in events if e.type == "tool_result").result["error"]
    assert any(e.type == "plan_ready" for e in events)


# --------------------------------------------------------------------------- #
# spilling
# --------------------------------------------------------------------------- #

async def test_large_results_are_spilled_and_compacted(job):
    big = {"ok": True, "rows": [{"i": i, "v": "x" * 50} for i in range(500)]}
    model = ScriptedModel([_reply(calls=[_call("get_job_status")]), _reply("ok")])
    call, listing, _ = fake_tools({"get_job_status": big})
    agent = RoadAgent(client=model, call_tool=call, list_tools=listing)
    convo = Conversation(job.dir / "c.json")
    events = await run(agent, convo, "status", job_id=job.job_id)

    tool_msg = next(m for m in convo.messages if m["role"] == "tool")
    assert len(tool_msg["content"]) <= SPILL_CHARS
    assert "full copy at" in tool_msg["content"]
    spilled = list((job.dir / "spill").glob("*.json"))
    assert len(spilled) == 1 and len(json.loads(spilled[0].read_text())["rows"]) == 500
    assert "spilled" in next(e for e in events if e.type == "tool_result").summary


def test_compact_keeps_shape_and_scalars():
    out = compact({"a": 1, "xs": list(range(40)), "rows": [{"k": i} for i in range(9)]})
    assert out["a"] == 1
    assert out["xs"][-1] == "...28 more"
    assert len(out["rows"]) == 6 and out["rows"][-1] == "...4 more"
