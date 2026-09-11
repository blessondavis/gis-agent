"""The task rules: order, budgets, loops, measurement, keep-best, plateau, done."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from gisagent import pipeline
from gisagent.agent import rules
from gisagent.agent.loop import Conversation, RoadAgent

LAT, LNG = 42.35, -71.06


@pytest.fixture
def job(tmp_path, monkeypatch):
    """A labelled job: one 7 px road across a 400 x 400 m truth raster."""
    from pyproj import Transformer

    monkeypatch.setattr(pipeline, "jobs_root", lambda: tmp_path)
    j = pipeline.new_job("rules")
    x0, y0 = Transformer.from_crs(4326, 3857, always_xy=True).transform(LNG, LAT)
    tf = from_origin(x0, y0 + 400, 1.0, 1.0)
    truth = np.zeros((400, 400), np.uint8)
    truth[197:204, :] = 255
    prof = dict(driver="GTiff", width=400, height=400, count=1, dtype="uint8",
                crs="EPSG:3857", transform=tf)
    with rasterio.open(j.truth_path, "w", **prof) as dst:
        dst.write(truth, 1)
    with rasterio.open(j.conf_path, "w", **{**prof, "dtype": "float32"}) as dst:
        dst.write((truth > 0).astype("float32"), 1)
    j.mask_path.write_bytes(b"")
    j.chips_dir.mkdir()
    (j.chips_dir / "chips.json").write_text("[]")
    j.conf_dir.mkdir()
    (j.conf_dir / "c.npy").write_bytes(b"")
    inv = Transformer.from_crs(3857, 4326, always_xy=True).transform
    j.road = lambda xs, row: [list(inv(x0 + x, y0 + 400 - row)) for x in xs]
    return j


def write_roads(job, *lines):
    fc = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {"confidence": 0.9},
         "geometry": {"type": "LineString", "coordinates": c}} for c in lines]}
    job.vector_path.write_text(json.dumps(fc))
    job.rebuild_network()


def half(job):
    return job.road([0, 200], 200.5)


def full(job):
    return job.road([0, 399], 200.5)


def wrong(job):
    return job.road([0, 399], 60)


# --------------------------------------------------------------------------- #
# per-call rules
# --------------------------------------------------------------------------- #

def test_calls_out_of_order_are_refused_with_the_next_step(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "jobs_root", lambda: tmp_path)
    j = pipeline.new_job("fresh")
    led = rules.Ledger()
    assert "tile_region" in rules.precheck("segment_chips", {}, j, led, task=True)
    assert "segment_chips" in rules.precheck("stitch_result", {}, j, led, task=True)
    assert "stitch_result" in rules.precheck("vectorize_result", {}, j, led, task=True)
    assert "try_candidates first" in rules.precheck("apply_candidate", {}, j, led, task=True)
    assert "no ground truth" in rules.precheck("evaluate_result", {}, j, led, task=True)
    assert rules.precheck("tile_region", {}, j, led, task=True) is None


def test_budgets_bind_in_a_task_but_not_in_conversation(job):
    led = rules.Ledger(spec=rules.TaskSpec(max_segment_runs=1))
    rules.note_call(led, "segment_chips", {"prompt": "a"}, ok=True)
    assert "budget spent" in rules.precheck("segment_chips", {"prompt": "b"}, job, led, task=True)
    assert rules.precheck("segment_chips", {"prompt": "b"}, job, led, task=False) is None
    # re-segmenting a few named chips is a rework, not a whole-region run
    assert rules.precheck("segment_chips", {"chip_ids": ["a"]}, job, led, task=True) is None


def test_reworks_measured_not_to_help_the_unet_are_refused(job):
    led = rules.Ledger()
    assert "upscale" in rules.precheck("segment_chips", {"upscale": 2, "backend": "unet"},
                                       job, led, task=False)
    native = rules.precheck("refine_area", {"area": "top", "backend": "unet"},
                            job, led, task=False)
    assert "cannot change anything" in native
    assert rules.precheck("refine_area", {"area": "top", "backend": "sam3", "upscale": 2},
                          job, led, task=False) is None


def test_a_bound_conversation_does_not_create_other_regions(job):
    why = rules.precheck("create_region_job", {"tile_names": ["a"]}, job, rules.Ledger(), task=True)
    assert job.job_id in why and "New region" in why


def test_stopping_needs_a_productive_lever_not_every_budget(job):
    """The live deadlock: candidates spent, the other budgets unusable with the
    U-Net, and "budget spent" demanded every budget -- so it never held."""
    led = rules.Ledger(spec=rules.TaskSpec(max_candidate_rounds=1))
    assert led.levers_left() == ["try_candidates (1 left)"]   # U-Net: no refine lever
    rules.note_call(led, "try_candidates", {"objective": "balanced"}, ok=True)
    assert led.levers_left() == [] and led.budget_spent()


async def test_an_agent_that_stops_acting_is_not_sent_back_forever(job):
    write_roads(job, full(job))

    async def tools(name, args):
        return {"ok": True}

    async def listing():
        return []

    model = Model([_reply("done")] * 6)
    agent = RoadAgent(client=model, call_tool=tools, list_tools=listing, max_steps=10)
    events = [e async for e in agent.chat(Conversation(job.dir / "c.json"), "go",
                                          job_id=job.job_id, mode="task")]
    stops = [e for e in events if e.type == "verify" and "no further progress" in e.text]
    assert len(stops) == 1 and len(model.replies) >= 3        # gave up early, not at 8 gates
    assert next(e for e in events if e.type == "report").result["status"] == "incomplete"


def test_the_same_call_on_the_same_state_is_refused(job):
    led = rules.Ledger()
    args = {"threshold": 0.5}
    rules.note_call(led, "stitch_result", args, ok=True, state=rules.state_signature(job))
    assert "exactly these arguments" in rules.precheck("stitch_result", args, job, led, task=True)
    job.mask_path.write_bytes(b"changed")            # the state moved on
    assert rules.precheck("stitch_result", args, job, led, task=True) is None


# --------------------------------------------------------------------------- #
# measuring, keep-best, plateau, done
# --------------------------------------------------------------------------- #

def test_the_harness_measures_against_truth_and_keeps_the_best(job):
    led = rules.Ledger(spec=rules.TaskSpec(objective="precision"))
    write_roads(job, half(job))
    m1 = rules.record(job, led, "first")
    assert m1.basis == "ground truth" and m1.precision > 0.95 and 0.4 < m1.recall < 0.6
    write_roads(job, full(job))
    m2 = rules.record(job, led, "better")
    assert m2.best and m2.score > m1.score
    write_roads(job, full(job), wrong(job))
    m3 = rules.record(job, led, "worse")
    assert not m3.best and m3.score < m2.score
    assert any("restore_best" in p for p in rules.unmet(job, led))
    rules.restore_best(job, led)
    assert rules.measure(job, led.spec)["score"] == pytest.approx(m2.score)


def test_plateau_is_two_changes_without_a_gain(job):
    led = rules.Ledger(spec=rules.TaskSpec(plateau_window=2))
    write_roads(job, half(job))
    rules.record(job, led, "a")
    write_roads(job, full(job))
    rules.record(job, led, "b")
    assert not led.plateaued()
    write_roads(job, full(job), wrong(job))
    rules.record(job, led, "c")
    assert not led.plateaued()                       # b is still inside the window
    write_roads(job, full(job))
    rules.record(job, led, "d")
    assert led.plateaued()


def test_definition_of_done_and_verdict(job):
    led = rules.Ledger(spec=rules.TaskSpec(target_precision=0.9, target_recall=0.9))
    assert rules.verdict(job, led)[0] == "failed"
    write_roads(job, full(job))
    rules.record(job, led, "full")
    left = " ".join(rules.unmet(job, led))
    assert "try_candidates" in left and "suggest_missing_roads" in left
    led.candidate_versions.append(rules.confidence_version(job))
    led.suggested_after = len(led.measurements) - 1
    assert rules.unmet(job, led) == []
    assert rules.verdict(job, led) == ("complete", "targets met")
    rep = rules.write_report(job, led)
    assert rep["status"] == "complete"
    assert "# Annotation report: complete" in (job.dir / "report.md").read_text()


def test_a_rework_that_changes_the_map_cannot_erase_a_better_result(job):
    """The live-run bug: without labels, re-running the model changes the
    confidence map the judge scores against. Two earlier rules lost a better
    network here (a fresh series; re-scoring under the new map, which favours
    the network drawn from it). Now the new network must win under the best
    result's map, which is biased against it."""
    job.truth_path.unlink()                          # the unlabelled case
    led = rules.Ledger(spec=rules.TaskSpec(objective="precision"))
    write_roads(job, full(job))
    first = rules.record(job, led, "apply_candidate")
    assert first.best and first.basis == "expected"

    # a rework: the model now also believes in a spurious road at row 60 ...
    with rasterio.open(job.conf_path, "r+") as ds:
        a = ds.read(1)
        a[57:64, :] = 0.6
        ds.write(a, 1)
    # ... and the network follows it
    write_roads(job, full(job), wrong(job))
    worse = rules.record(job, led, "refine_area")
    assert worse.version != first.version
    assert not worse.best                            # old rule: this became "best"
    assert led.best is first and led.worse_than_best()
    assert any("restore_best" in p for p in rules.unmet(job, led))

    rules.restore_best(job, led)
    back = rules.record(job, led, "restore_best")
    assert back.version == first.version and back.score == pytest.approx(first.score)


def test_confidence_version_is_by_content_not_timestamp(job):
    v = rules.confidence_version(job)
    data = job.conf_path.read_bytes()
    job.conf_path.write_bytes(data)                  # rewritten, identical
    assert rules.confidence_version(job) == v
    job.conf_path.write_bytes(data[:-1] + b"\x01")
    assert rules.confidence_version(job) != v


# --------------------------------------------------------------------------- #
# a whole autonomous task, scripted
# --------------------------------------------------------------------------- #

async def test_the_harness_binds_job_id_whatever_the_model_types(job):
    got = []

    async def tools(name, args):
        got.append(args["job_id"])
        return {"ok": True}

    async def listing():
        return [SimpleNamespace(name="get_job_status", description="",
                                input_schema={"type": "object",
                                              "properties": {"job_id": {"type": "string"}}})]

    # a live run really did mistype a 22-character id like this
    model = Model([_reply(calls=[_call("get_job_status", {"job_id": job.job_id[:-3] + "400"})]),
                   _reply("ok")])
    agent = RoadAgent(client=model, call_tool=tools, list_tools=listing)
    _ = [e async for e in agent.chat(Conversation(job.dir / "c.json"), "status?",
                                     job_id=job.job_id)]
    assert got == [job.job_id]

def _call(name, args=None):
    return SimpleNamespace(id=f"c_{name}_{id(args)}",
                           function=SimpleNamespace(name=name, arguments=json.dumps(args or {})))


def _reply(content="", calls=None):
    return SimpleNamespace(choices=[SimpleNamespace(
        message=SimpleNamespace(content=content, tool_calls=calls or None))])


class Model:
    def __init__(self, replies):
        self.replies = list(replies)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, **kw):
        return self.replies.pop(0)


async def test_an_autonomous_task_is_held_to_the_rules(job):
    seen = {}

    async def tools(name, args):
        seen.setdefault(name, []).append(args)
        if name == "vectorize_result":
            write_roads(job, half(job))
        elif name == "try_candidates":
            (job.dir / "candidates").mkdir(exist_ok=True)
            (job.dir / "candidates" / "index.json").write_text(json.dumps(
                {"candidates": [{"id": "c1", "score": 0.0, "params": {}}]}))
        elif name == "apply_candidate":
            write_roads(job, full(job))
        elif name == "refine_area":
            write_roads(job, full(job), wrong(job))
        return {"ok": True}

    async def listing():
        return [SimpleNamespace(name=n, description=n, input_schema={"type": "object"})
                for n in ("vectorize_result", "try_candidates", "apply_candidate",
                          "refine_area", "suggest_missing_roads", "segment_chips")]

    model = Model([
        _reply(calls=[_call("vectorize_result")]),
        _reply("All done."),                                          # 1. too early
        _reply(calls=[_call("try_candidates", {"objective": "recall"})]),
        _reply(calls=[_call("apply_candidate", {"candidate_id": "c1"})]),
        _reply(calls=[_call("refine_area", {"area": "top", "backend": "sam3"})]),  # worse
        _reply("Done now."),                                          # 2. worse than best
        _reply(calls=[_call("restore_best")]),
        _reply(calls=[_call("vectorize_result")]),                    # 3. plateau: refused
        _reply(calls=[_call("suggest_missing_roads")]),
        _reply("Finished: the best network is restored and handed off."),
    ])
    spec = rules.TaskSpec(objective="precision")
    agent = RoadAgent(client=model, call_tool=tools, list_tools=listing, max_steps=20)
    events = [e async for e in agent.chat(Conversation(job.dir / "c.json"), "annotate",
                                          job_id=job.job_id, mode="task", spec=spec)]

    gates = [e.text for e in events if e.type == "verify" and not e.ok]
    assert "try_candidates" in gates[0] and "suggest_missing_roads" in gates[0]
    assert "restore_best" in gates[1]
    # the model asked for recall; the task's objective was used
    assert seen["try_candidates"][0]["objective"] == "precision"
    refusals = [e.text for e in events if e.type == "rule"]
    assert any("plateau" in r for r in refusals)
    measures = [e.result for e in events if e.type == "measure"]
    assert [m["best"] for m in measures][:2] == [True, True] and not measures[2]["best"]
    report = next(e for e in events if e.type == "report").result
    assert report["status"] == "best_effort" and "plateau" in report["reason"]
    assert report["final"]["score"] == pytest.approx(report["best"]["score"])
    assert (job.dir / "report.json").exists() and events[-1].type == "done"
