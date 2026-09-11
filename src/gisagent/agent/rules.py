"""The rules of the annotation task, enforced by the harness rather than hoped for.

An agent that is meant to finish a job unattended cannot be steered by prompt
text alone: a model will skip a step, repeat a failing call, stop at the first
plausible result, or declare success on a number it misremembered. So every
rule here exists twice -- once as text the model reads, once as code the
harness runs on every tool call and before every attempt to finish:

* **Order.** The pipeline has prerequisites; a call without them is refused
  with the step that is actually next, before any GPU time is spent.
* **The harness measures.** After every change to the network the harness
  scores it itself -- against ground truth when there is some, else by the
  network's expected precision/recall under the model's confidence (the judge
  validated in docs/harness.md). The model reads the score; it never has to
  remember to compute one.
* **Keep the best.** The best-scoring result is snapshotted. The agent may not
  finish below it, and ``restore_best`` brings it back.
* **Budgets** on the expensive and destructive calls, and **no loops**: an
  identical call repeated is refused.
* **A stop condition decided by code**: targets met, a plateau (the last few
  changes gained nothing), or the budget spent.
* **A definition of done** that must hold before the agent may finish, and a
  **report** whose verdict comes from the checks, not from the model's
  summary.

Label-free scores are only comparable within one confidence map: re-running
the model changes the map the judge scores against. Each measurement is
tagged with the map's version, and keep-best and plateau compare like with
like. With ground truth every measurement is comparable.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

OBJECTIVES = ("balanced", "precision", "recall")

# Tools that change the live result (and therefore get measured).
CHANGES_RESULT = frozenset({
    "segment_chips", "stitch_result", "vectorize_result", "refine_area",
    "repair_geometry", "apply_candidate", "restore_best",
})
# Calls that cost minutes of GPU or overwrite model output: budgeted.
BUDGETED = {"segment_chips": "max_segment_runs", "refine_area": "max_refines",
            "try_candidates": "max_candidate_rounds"}
# What "trying to improve" means, for the plateau rule.
IMPROVING = frozenset({
    "segment_chips", "stitch_result", "vectorize_result", "refine_area",
    "repair_geometry", "apply_candidate", "try_candidates",
})


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --------------------------------------------------------------------------- #
# the task
# --------------------------------------------------------------------------- #

@dataclass
class TaskSpec:
    """What the person asked for, and the limits the agent works within."""

    objective: str = "balanced"          # balanced | precision | recall
    target_precision: float | None = None   # only meaningful with ground truth
    target_recall: float | None = None
    max_segment_runs: int = 2            # whole-region model runs
    max_refines: int = 4                 # area reworks
    max_candidate_rounds: int = 3
    plateau_window: int = 2              # changes in a row without a gain
    min_gain: float = 0.003              # a gain smaller than this is noise

    def __post_init__(self) -> None:
        if self.objective not in OBJECTIVES:
            raise ValueError(f"objective must be one of {OBJECTIVES}")

    @property
    def beta(self) -> float:
        return {"balanced": 1.0, "precision": 0.5, "recall": 2.0}[self.objective]

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- #
# the ledger: what happened in this task, persisted with the job
# --------------------------------------------------------------------------- #

@dataclass
class Measurement:
    at: str
    after: str                 # the tool call that produced this state
    score: float
    precision: float
    recall: float
    basis: str                 # "ground truth" | "expected"
    version: str               # which confidence map (label-free comparability)
    km: float
    n_features: int
    topology: float
    best: bool = False


@dataclass
class Ledger:
    spec: TaskSpec = field(default_factory=TaskSpec)
    started_at: str = field(default_factory=_utc)
    measurements: list[Measurement] = field(default_factory=list)
    calls: list[dict] = field(default_factory=list)      # {tool, args_key, ok}
    candidate_versions: list[str] = field(default_factory=list)
    suggested_after: int = -1          # index of the measurement suggestions followed
    best_index: int = -1
    last_compare: list | None = None   # [live, best] on the same basis, at the last measurement
    turn_start: int = 0                # first call of the current turn (not persisted)

    # -- persistence ---------------------------------------------------------- #

    @staticmethod
    def path(job) -> Path:
        return job.dir / "task.json"

    @classmethod
    def load(cls, job) -> "Ledger | None":
        p = cls.path(job)
        if not p.exists():
            return None
        d = json.loads(p.read_text(encoding="utf-8"))
        led = cls(spec=TaskSpec(**d["spec"]), started_at=d["started_at"])
        led.measurements = [Measurement(**m) for m in d["measurements"]]
        led.calls = d["calls"]
        led.candidate_versions = d.get("candidate_versions", [])
        led.suggested_after = d.get("suggested_after", -1)
        led.best_index = d.get("best_index", -1)
        led.last_compare = d.get("last_compare")
        return led

    def save(self, job) -> None:
        d = {"spec": self.spec.to_dict(), "started_at": self.started_at,
             "measurements": [asdict(m) for m in self.measurements],
             "calls": self.calls[-400:], "candidate_versions": self.candidate_versions,
             "suggested_after": self.suggested_after, "best_index": self.best_index,
             "last_compare": self.last_compare}
        tmp = self.path(job).with_suffix(".tmp")
        tmp.write_text(json.dumps(d, indent=1), encoding="utf-8")
        tmp.replace(self.path(job))

    # -- queries --------------------------------------------------------------- #

    def count(self, tool: str) -> int:
        return sum(1 for c in self.calls if c["tool"] == tool and c["ok"])

    @property
    def latest(self) -> Measurement | None:
        return self.measurements[-1] if self.measurements else None

    @property
    def best(self) -> Measurement | None:
        return self.measurements[self.best_index] if self.best_index >= 0 else None

    def comparable(self, a: Measurement, b: Measurement) -> bool:
        return a.basis == "ground truth" == b.basis or a.version == b.version

    def plateaued(self) -> bool:
        """The last ``plateau_window`` changes produced no new best.

        "New best" is decided in :func:`record` on a like-for-like basis, so
        this holds with or without ground truth.
        """
        w = self.spec.plateau_window
        if len(self.measurements) <= w:
            return False
        return not any(m.best for m in self.measurements[-w:])

    def worse_than_best(self) -> bool:
        """The live result lost the last like-for-like comparison with the best."""
        if self.best is None or self.latest is None or self.latest is self.best:
            return False
        cand, ref = self.last_compare or (None, None)
        if cand is None or ref is None:
            return False
        return cand < ref - self.spec.min_gain

    def targets_met(self) -> bool:
        s, cur = self.spec, self.latest
        if cur is None or cur.basis != "ground truth":
            return False
        if s.target_precision is None and s.target_recall is None:
            return False
        return ((s.target_precision is None or cur.precision >= s.target_precision)
                and (s.target_recall is None or cur.recall >= s.target_recall))

    def budget_spent(self) -> bool:
        """No *productive* way left to change the result.

        Not "every budget is used up": a lever the rules refuse, or one known
        not to help, is not a way forward. The first version counted all
        budgets, and a live run deadlocked -- candidates spent, the remaining
        budgets unusable with the U-Net, no stop condition reachable.
        """
        return not self.levers_left()

    def levers_left(self) -> list[str]:
        """Ways to change the result that the rules allow and that can help."""
        from gisagent.config import get_settings

        s = self.spec
        left = []
        n = s.max_candidate_rounds - self.count("try_candidates")
        if n > 0:
            left.append(f"try_candidates ({n} left)")
        # the U-Net is deterministic at native scale and upscaling it hurts,
        # so area reworks and re-runs only count with another backend
        if (get_settings().segment_backend or "unet").lower() != "unet":
            n = s.max_refines - self.count("refine_area")
            if n > 0:
                left.append(f"refine_area ({n} left)")
        return left


# --------------------------------------------------------------------------- #
# measuring (the harness's own, deterministic)
# --------------------------------------------------------------------------- #

_VERSION_CACHE: dict[tuple, str] = {}


def confidence_version(job) -> str:
    """Identity of the confidence map, by content.

    Not by mtime: apply_candidate re-stitches, rewriting an identical file,
    and a timestamp would call that a new map and demand a new candidate
    round every time. Hashing ~36 MB takes ~0.1 s, and is cached by stat.
    """
    import hashlib

    try:
        st = job.conf_path.stat()
    except FileNotFoundError:
        return "none"
    key = (str(job.conf_path), st.st_size, st.st_mtime_ns)
    if key not in _VERSION_CACHE:
        h = hashlib.blake2b(digest_size=8)
        with job.conf_path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        _VERSION_CACHE[key] = h.hexdigest()
    return _VERSION_CACHE[key]


def measure(job, spec: TaskSpec) -> dict | None:
    """Score the live network under the task's objective. CPU only, a few s."""
    import numpy as np
    import rasterio

    from gisagent.evaluate.network import expected_scores, f_beta, score_network
    from gisagent.vector.topology import analyse

    if not job.vector_path.exists():
        return None
    net = job.network()
    if not net.get("features"):
        return {"score": 0.0, "precision": 0.0, "recall": 0.0, "basis": "empty",
                "version": confidence_version(job), "km": 0.0, "n_features": 0,
                "topology": 0.0}
    if job.has_truth():
        s = score_network(net, job.truth_path)["overall"]
        p, r, basis = s["correctness"], s["completeness"], "ground truth"
    elif job.conf_path.exists():
        with rasterio.open(job.conf_path) as ds:
            conf = ds.read(1).astype(np.float32)
            e = expected_scores(net, conf, ds.transform, ds.crs)
        p, r, basis = e["precision"], e["recall"], "expected"
    else:
        return None
    stats = net.get("stats", {})
    return {"score": round(f_beta(p, r, spec.beta), 4), "precision": round(p, 4),
            "recall": round(r, 4), "basis": basis, "version": confidence_version(job),
            "km": stats.get("total_length_km", 0.0),
            "n_features": stats.get("n_features", len(net["features"])),
            "topology": round(analyse(net).score, 4)}


def record(job, ledger: Ledger, after: str) -> Measurement | None:
    """Measure the live result, add it to the ledger, snapshot it if best."""
    m = measure(job, ledger.spec)
    if m is None:
        return None
    meas = Measurement(at=_utc(), after=after, **m)
    ledger.measurements.append(meas)
    best = ledger.best
    if best is None:
        cand, ref = meas.score, None
    elif ledger.comparable(meas, best):
        cand, ref = meas.score, best.score
    else:
        # The model was re-run, so the confidence map changed, and the label-
        # free judge would be comparing networks under different maps. Judge
        # the new network under the *best's* map: that map is biased towards
        # the old network, so a new one that still wins there is better.
        #
        # Measured, not assumed (docs/harness.md): over eight area reworks on
        # Boston, all of which truly hurt, judging under the new map said
        # "keep" 8/8; judging under the old map said "reject" 8/8. The first
        # two versions of this rule got it wrong -- one started a fresh series,
        # one re-scored under the new map -- and a live run lost a better
        # network to each.
        cand, ref = score_under_best_map(job, ledger), best.score
    ledger.last_compare = [cand, ref]
    if ref is None or (cand is not None and cand > ref + ledger.spec.min_gain):
        ledger.best_index = len(ledger.measurements) - 1
        meas.best = True
        _snapshot_best(job)
    ledger.save(job)
    return meas


def score_under_best_map(job, ledger: Ledger) -> float | None:
    """The live network, scored under the confidence map of the best result."""
    import numpy as np
    import rasterio

    from gisagent.evaluate.network import expected_scores, f_beta

    snap = job.dir / "best" / "confidence.tif"
    if not snap.exists():
        return None
    with rasterio.open(snap) as ds:
        conf = ds.read(1).astype(np.float32)
        e = expected_scores(job.network(), conf, ds.transform, ds.crs)
    return round(f_beta(e["precision"], e["recall"], ledger.spec.beta), 4)


BEST_FILES = ("mask.tif", "roads.geojson", "confidence.tif")


def _snapshot_best(job) -> None:
    d = job.dir / "best"
    d.mkdir(exist_ok=True)
    for name in BEST_FILES:
        src = job.dir / name
        if src.exists():
            # through the turn's checkpoint, so a rewind also rewinds "best"
            job._preserve(d / name)
            shutil.copy2(src, d / name)       # copy2 keeps mtime -> same version


def restore_best(job, ledger: Ledger) -> dict:
    d = job.dir / "best"
    best = ledger.best
    if best is None or not d.is_dir():
        raise RuntimeError("nothing measured yet, so there is no best result to restore")
    for name in BEST_FILES:
        if (d / name).exists():
            shutil.copy2(d / name, job.dir / name)
    job.rebuild_network()
    return {"restored": best.after, "score": best.score,
            "note": "mask, centrelines and confidence map are back as they were at "
                    "the best measurement; per-chip confidences are not, so a later "
                    "stitch_result would recombine the current ones"}


# --------------------------------------------------------------------------- #
# the rules on each call
# --------------------------------------------------------------------------- #

def state_signature(job) -> str:
    """Changes whenever the model output, mask or network changes."""
    if job is None:
        return ""
    parts = []
    for p in (job.mask_path, job.vector_path, job.edits_path,
              job.dir / "candidates" / "index.json"):
        try:
            st = p.stat()
            parts.append(f"{st.st_size}.{st.st_mtime_ns}")
        except FileNotFoundError:
            parts.append("-")
    if job.conf_dir.exists():
        parts.append(str(max((q.stat().st_mtime_ns for q in job.conf_dir.glob("*.npy")),
                             default=0)))
    return "|".join(parts)


def _key(tool: str, args: dict, state: str = "") -> str:
    """A call is a repeat only if the arguments AND the job's state are the same:
    vectorising again after a rework is legitimate, re-running it on an
    unchanged mask is not."""
    a = {k: v for k, v in (args or {}).items() if k != "job_id"}
    return tool + json.dumps(a, sort_keys=True, default=str) + "@" + state


def precheck(tool: str, args: dict, job, ledger: Ledger | None, *, task: bool) -> str | None:
    """Why this call must not run now, or None. Cheap: file checks only."""
    if job is None:
        return None
    if tool == "create_region_job":
        # a live task created a second region from other tiles; its later calls
        # were bound to the right job, but it left an orphan behind
        return (f"this conversation is bound to job {job.job_id}, whose region already "
                "exists: work on it. If the person wants a different region, tell them to "
                "create it (New region in the web app, or `gisagent build`)")
    chips = (job.chips_dir / "chips.json").exists()
    chip_conf = job.conf_dir.exists() and any(job.conf_dir.glob("*.npy"))
    order = {
        "segment_chips": (chips, "tile the region first: tile_region"),
        "stitch_result": (chip_conf, "there is nothing to stitch: segment_chips first"),
        "vectorize_result": (job.mask_path.exists(), "there is no mask yet: stitch_result first"),
        "try_candidates": (job.conf_path.exists(),
                           "candidates need a stitched confidence map: stitch_result first"),
        "refine_area": (chips and chip_conf, "rework needs a segmented region: segment_chips first"),
        "apply_candidate": ((job.dir / "candidates" / "index.json").exists(),
                            "there are no candidates: try_candidates first"),
        "check_topology": (job.vector_path.exists(), "there is no network yet: vectorize_result first"),
        "suggest_missing_roads": (job.conf_path.exists() and job.vector_path.exists(),
                                  "suggestions need a network and a confidence map"),
    }
    if tool in order and not order[tool][0]:
        return order[tool][1]
    if tool in ("evaluate_result", "sweep_threshold") and not job.has_truth():
        return ("this job has no ground truth. The harness already scores every change "
                "without labels; use check_topology for a structural check")
    if tool in ("segment_chips", "refine_area"):
        why = _unet_rework_problem(tool, args)
        if why:
            return why
    if ledger is None:
        return None

    # budgets: a task's limits. In conversation the person is the budget.
    if task and tool in BUDGETED and not (tool == "segment_chips" and args.get("chip_ids")):
        cap = getattr(ledger.spec, BUDGETED[tool])
        if ledger.count(tool) >= cap:
            return (f"budget spent: {tool} has run {cap} time(s), the limit for this "
                    "task. Work with what you have, or finish")
    # loops: the same call on the same state (in conversation, this turn only)
    k = _key(tool, args, state_signature(job))
    same = sum(1 for c in ledger.calls[ledger.turn_start:] if c["key"] == k)
    limit = 1 if tool in CHANGES_RESULT or tool == "try_candidates" else 2
    if same >= limit:
        last = next((c for c in reversed(ledger.calls) if c["key"] == k), None)
        why = "it failed last time" if last and not last["ok"] else "nothing has changed since"
        return (f"refused: {tool} was already called with exactly these arguments and {why}. "
                "Change something, or move on")
    # plateau: stop improving, go and finish
    if task and tool in IMPROVING and ledger.plateaued():
        return ("the result has plateaued: the last changes gained nothing under the task's "
                "objective. Stop improving; restore_best if the live result is not the "
                "best, then hand off with suggest_missing_roads and finish")
    return None


def _unet_rework_problem(tool: str, args: dict) -> str | None:
    """Model re-runs that are known not to help the U-Net.

    Both facts are measured. Upscaling: on Boston, eight area reworks at
    upscale 2 all made the true score worse (0.81 -> 0.45-0.74), and a scale
    sweep on Global-Scale found native scale best. At native scale the U-Net
    is deterministic, so an area rework returns the prediction it already has.
    """
    from gisagent.config import get_settings

    backend = (args.get("backend") or get_settings().segment_backend or "unet").lower()
    if backend != "unet":
        return None
    up = int(args.get("upscale") or 1)
    if up > 1:
        return ("upscale > 1 with the U-Net is refused: it was trained at 1 m/px, and on "
                "Boston every upscaled rework measured made the true score worse. Use "
                "upscale 1")
    if tool == "refine_area":
        return ("a U-Net rework at native scale returns the prediction the chips already "
                "have, so it cannot change anything. To improve, run try_candidates with "
                "other vectoriser settings (close_radius, min_object_px, min_length_m), "
                "or rework with backend='sam3' for a different model's view")
    return None


def note_call(ledger: Ledger | None, tool: str, args: dict, ok: bool,
              refused: bool = False, state: str = "") -> None:
    """Record a call. ``state`` is the job's signature *before* the call, so a
    later identical call on the same state is recognised as a repeat."""
    if ledger is not None:
        ledger.calls.append({"tool": tool, "key": _key(tool, args, state), "ok": ok,
                             "refused": refused, "at": _utc()})


# --------------------------------------------------------------------------- #
# the definition of done
# --------------------------------------------------------------------------- #

def unmet(job, ledger: Ledger) -> list[str]:
    """What must still happen before the task may finish. Empty means done."""
    problems = []
    cur = ledger.latest
    if not job.vector_path.exists() or cur is None:
        return ["there is no measured road network yet: run the pipeline through "
                "vectorize_result (tile_region -> segment_chips -> stitch_result -> "
                "vectorize_result)"]
    if cur.n_features == 0:
        problems.append("the network is empty: check the mask threshold (try_candidates)")
    if cur.version not in ledger.candidate_versions:
        problems.append(
            f"the operating point was not chosen for the current confidence map: run "
            f"try_candidates (objective {ledger.spec.objective!r}) and apply_candidate the best")
    else:
        cand = _best_candidate(job)
        if cand and cand["score"] > cur.score + ledger.spec.min_gain and cur.basis == (
                "ground truth" if job.has_truth() else "expected"):
            problems.append(f"candidate {cand['id']} scores {cand['score']:.3f} but the live "
                            f"result scores {cur.score:.3f}: apply_candidate {cand['id']}")
    if ledger.worse_than_best():
        live, ref = ledger.last_compare
        problems.append(f"the live result is worse than the best measured ({live:.3f} vs "
                        f"{ref:.3f}, compared like for like; best was after "
                        f"{ledger.best.after}): restore_best")
    if not (ledger.targets_met() or ledger.plateaued() or ledger.budget_spent()):
        problems.append(_why_not_stopping(ledger))
    if ledger.suggested_after < len(ledger.measurements) - 1:
        problems.append("hand off: run suggest_missing_roads on the final network so the "
                        "person gets a review queue")
    return problems


def _why_not_stopping(ledger: Ledger) -> str:
    s = ledger.spec
    target = ""
    if s.target_precision is not None or s.target_recall is not None:
        target = " The targets are not met yet."
    return ("no stop condition holds yet: the result is still improving and useful "
            f"budget remains ({', '.join(ledger.levers_left())}).{target} Try a real change "
            "-- another try_candidates round that varies the vectoriser settings around the "
            "best so far -- until two changes in a row gain nothing")


def _best_candidate(job) -> dict | None:
    p = job.dir / "candidates" / "index.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))["candidates"][0]
    except (json.JSONDecodeError, KeyError, IndexError):
        return None


# --------------------------------------------------------------------------- #
# the report
# --------------------------------------------------------------------------- #

def verdict(job, ledger: Ledger) -> tuple[str, str]:
    left = unmet(job, ledger)
    if ledger.latest is None:
        if job.vector_path.exists():
            return "incomplete", "a network exists but this task never measured a change to it"
        return "failed", "no road network was produced"
    if left:
        return "incomplete", "; ".join(left)
    if ledger.targets_met():
        return "complete", "targets met"
    if ledger.plateaued():
        return "best_effort", "stopped at a plateau: further changes gained nothing"
    return "best_effort", "stopped when no useful budget was left"


def write_report(job, ledger: Ledger, *, duration_s: float = 0.0) -> dict:
    status, reason = verdict(job, ledger)
    cur, best = ledger.latest, ledger.best
    s = ledger.spec
    try:
        sugg = json.loads(job.suggestions_path.read_text(encoding="utf-8"))["stats"]
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        sugg = None
    rep = {
        "status": status, "reason": reason, "objective": s.objective,
        "basis": cur.basis if cur else None,
        "final": asdict(cur) if cur else None,
        "best": asdict(best) if best else None,
        "history": [{k: getattr(m, k) for k in ("after", "score", "precision", "recall", "km", "best")}
                    for m in ledger.measurements],
        "budgets": {t: f"{ledger.count(t)}/{getattr(s, k)}" for t, k in BUDGETED.items()},
        "refused_calls": sum(1 for c in ledger.calls if c.get("refused")),
        "review_queue": sugg,
        "spec": s.to_dict(), "duration_s": round(duration_s, 1), "at": _utc(),
    }
    (job.dir / "report.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    (job.dir / "report.md").write_text(_markdown(rep), encoding="utf-8")
    return rep


def _markdown(r: dict) -> str:
    f = r["final"] or {}
    label = {"precision": "correctness", "recall": "completeness"}
    what = "measured against ground truth" if r["basis"] == "ground truth" else \
        "expected under the model's confidence (no ground truth)"
    lines = [
        f"# Annotation report: {r['status'].replace('_', ' ')}",
        "", f"{r['reason']}.", "",
        f"- objective: **{r['objective']}**, {what}",
        f"- final score {f.get('score', 0):.3f} (precision / {label['precision']} "
        f"{f.get('precision', 0):.3f}, recall / {label['recall']} {f.get('recall', 0):.3f})",
        f"- network: {f.get('km', 0)} km in {f.get('n_features', 0)} centrelines",
        "- budgets used: " + ", ".join(f"{k} {v}" for k, v in r["budgets"].items()),
    ]
    if r["review_queue"]:
        q = r["review_queue"]
        lines.append(f"- left for a person: {q.get('gap', 0)} gap connectors and "
                     f"{q.get('missed', 0)} possible roads in the review queue")
    lines += ["", "| step | after | score | precision | recall | km |", "|---|---|---|---|---|---|"]
    for i, h in enumerate(r["history"], 1):
        star = " (best)" if h["best"] else ""
        lines.append(f"| {i} | {h['after']}{star} | {h['score']:.3f} | {h['precision']:.3f} | "
                     f"{h['recall']:.3f} | {h['km']} |")
    return "\n".join(lines) + "\n"


def rules_text(spec: TaskSpec, *, labelled: bool) -> str:
    """The rules, as the model reads them. Every one is also enforced."""
    target = ""
    if labelled and (spec.target_precision or spec.target_recall):
        target = (f"\n- Targets: correctness >= {spec.target_precision or '-'}, "
                  f"completeness >= {spec.target_recall or '-'}. Meeting them ends the task.")
    basis = ("against ground truth" if labelled else
             "by expected precision/recall under the model's confidence (no labels)")
    return f"""

AUTONOMOUS TASK. You are annotating this region on your own. The harness
enforces every rule below: a call that breaks one is refused with the reason,
and you cannot finish until the definition of done holds.

Objective: {spec.objective!r} (F-beta, beta={spec.beta:g}). The harness scores
the network {basis} after every change and tells you the score. Do not
compute your own; quote the harness's.

Order (each step needs the one before):
  1. tile_region                     (skip if chips exist)
  2. segment_chips                   (U-Net unless told otherwise)
  3. stitch_result, vectorize_result
  4. try_candidates with objective {spec.objective!r}, then apply_candidate the best
  5. improve with further try_candidates rounds that vary the vectoriser too
     (close_radius 1-4, min_object_px 200-1500, min_length_m 15-60), around the
     best threshold so far. That is the lever that works with the U-Net: area
     reworks cannot help it (refused, with the measurements). refine_area is for
     backend='sam3'. Keep what helps; restore_best what does not.
  6. hand off: suggest_missing_roads on the final network
  7. finish with a short report: final score, what changed it, what is left

Budgets: segment_chips {spec.max_segment_runs} whole-region run(s), refine_area {spec.max_refines},
try_candidates {spec.max_candidate_rounds}. An identical call is never repeated.

Stop when one holds (the harness decides): targets met; plateau ({spec.plateau_window}
changes in a row gaining < {spec.min_gain}); or budget spent.{target}

Done means: a measured network exists; the operating point was chosen with
try_candidates on the current confidence map; the live result is the best
measured (restore_best otherwise); suggest_missing_roads has run on it.
"""
