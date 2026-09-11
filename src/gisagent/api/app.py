"""FastAPI backend: REST for the pipeline, WebSocket for the live agent trace.

No model is ever loaded in this process. Every operation that needs the GPU goes
through the shared MCP server, which owns the single resident copy of SAM 3.
That keeps VRAM within budget on one card and means the agent and the
deterministic baseline exercise exactly the same tools, so their numbers are
comparable.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import warnings
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from gisagent import mcp_client, pipeline
from gisagent.config import get_settings

warnings.filterwarnings("ignore", category=UserWarning)

WEB_DIR = Path(__file__).resolve().parents[3] / "web"


class RunRegistry:
    """Tracks in-flight agent turns and fans events out to WebSocket clients."""

    def __init__(self) -> None:
        self._subs: dict[str, set[asyncio.Queue]] = {}
        self._tasks: dict[str, asyncio.Task] = {}
        self._state: dict[str, str] = {}

    def subscribe(self, key: str) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=2000)
        self._subs.setdefault(key, set()).add(q)
        return q

    def unsubscribe(self, key: str, q: asyncio.Queue) -> None:
        subs = self._subs.get(key)
        if subs:
            subs.discard(q)
            if not subs:
                self._subs.pop(key, None)

    async def publish(self, key: str, event: dict) -> None:
        for q in list(self._subs.get(key, ())):
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                pass

    def state(self, key: str) -> str:
        return self._state.get(key, "idle")

    def set_state(self, key: str, value: str) -> None:
        self._state[key] = value

    def register(self, key: str, task: asyncio.Task) -> None:
        self._tasks[key] = task

    def cancel(self, key: str) -> bool:
        task = self._tasks.get(key)
        if task and not task.done():
            task.cancel()
            self.set_state(key, "cancelled")
            return True
        return False


registry = RunRegistry()


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_settings()
    yield
    await mcp_client.shutdown()


app = FastAPI(title="gis-agent", version="0.2.0", lifespan=lifespan)


# --------------------------------------------------------------------------- #
# models
# --------------------------------------------------------------------------- #

class CreateJobRequest(BaseModel):
    tile_names: list[str] = Field(default_factory=list)
    split: str = "train"
    name: str = ""
    block: int = 2
    screen: bool = True


class ChatRequest(BaseModel):
    message: str
    model: str | None = None
    max_steps: int | None = None
    mode: str = "work"            # work | plan (read-only; propose, then approve)


class RewindRequest(BaseModel):
    message_index: int            # rewind to just before this chat message


class PipelineRequest(BaseModel):
    chip_size: int = 1024
    overlap: int = 128
    prompt: str = "road network"
    seg_threshold: float = 0.3
    upscale: int = 1
    mask_threshold: float = 0.5


class EditRequest(BaseModel):
    """One edit to the network. Coordinates are WGS84 [[lng, lat], ...]."""

    op: str                                    # add | delete | replace | dismiss
    geometry: list[list[float]] | None = None  # the new line (add/replace/dismiss)
    target: dict | None = None                 # {source, id, geometry} (delete/replace)
    note: str = ""


class RefineRequest(BaseModel):
    bbox_wgs84: list[float] | None = None
    area: str = ""
    prompt: str = "road network"
    threshold: float = 0.25
    upscale: int = 2
    note: str = ""


# --------------------------------------------------------------------------- #
# meta
# --------------------------------------------------------------------------- #

@app.get("/api/health")
async def health() -> dict:
    from gisagent.qgis.process import available as qgis_available

    s = get_settings()
    try:
        import torch

        cuda = torch.cuda.is_available()
        gpu = torch.cuda.get_device_name(0) if cuda else None
    except Exception:
        cuda, gpu = False, None

    return {
        "ok": True,
        "llm_configured": s.llm_ready(),
        "llm_model": s.llm_model,
        "sam_model": s.sam_model,
        "sam_token": bool(s.hf_token),
        "qgis": qgis_available(),
        "cuda": cuda,
        "gpu": gpu,
        "device": s.device,
    }


@app.get("/api/tools")
async def tools() -> dict:
    try:
        listed = await mcp_client.list_tools()
    except Exception as exc:
        raise HTTPException(503, f"MCP server unavailable: {exc}")
    return {
        "tools": [
            {"name": t.name, "description": (t.description or "").strip()}
            for t in listed
        ]
    }


# --------------------------------------------------------------------------- #
# jobs
# --------------------------------------------------------------------------- #

@app.get("/api/jobs")
def jobs() -> dict:
    return {"jobs": pipeline.list_jobs()}


def _finish_job(job, names: list[str], split: str) -> dict:
    from gisagent.raster.preview import render_image_preview

    settings = get_settings()
    job.manifest["tiles"] = names
    job.manifest["split"] = split
    sat = [settings.raw_dir / "sat" / f"{n}.tif" for n in names]
    truth = [p for p in (settings.raw_dir / "map" / f"{n}.tif" for n in names)
             if p.exists()] or None
    region = job.build_region(sat, truth)
    render_image_preview(job.image_path, job.preview_dir / "image.png")
    return region


@app.post("/api/jobs")
async def create_job(req: CreateJobRequest) -> dict:
    from gisagent.dataset.mass_roads import (
        TileRef, decode_name, download_tiles, find_contiguous_block,
        list_split, measure_blank, rank_blocks,
    )
    from gisagent.raster.georef import georeference_labels

    settings = get_settings()
    MAX_BLANK = 0.12
    MAX_ATTEMPTS = 5

    def _download(refs):
        results = download_tiles(refs, settings.raw_dir)
        failed = [r.ref.name for r in results if not r.ok]
        if failed:
            raise HTTPException(502, f"tile download failed: {failed}")
        return results

    def _work() -> dict:
        names = req.tile_names
        quality: dict = {}

        if names:
            refs = [
                TileRef(name=n, split=req.split, key_e=decode_name(n)[0],
                        key_n=decode_name(n)[1])
                for n in names
            ]
            _download(refs)
        else:
            refs_all = list_split(req.split)
            if req.screen and req.block > 1:
                # Screening ranks on road density only; blankness cannot be
                # measured without the pixels, so walk the ranked candidates
                # and reject no-data blocks once their tiles are local.
                cache = settings.cache_dir / "tile_quality.json"
                scored, quality = rank_blocks(
                    refs_all, req.block, min_road=0.006,
                    search_limit=400, cache_path=cache,
                )
                if not scored:
                    raise HTTPException(400, "no suitable block found")
                for _road, block in scored[:MAX_ATTEMPTS]:
                    results = _download(block)
                    if max(measure_blank(r.image_path) for r in results) <= MAX_BLANK:
                        break
                else:
                    raise HTTPException(
                        400,
                        f"top {MAX_ATTEMPTS} candidate blocks were mostly no-data",
                    )
            else:
                block = find_contiguous_block(refs_all, req.block, req.block)
                if not block:
                    raise HTTPException(400, "no suitable block found")
                _download(block)
            names = [r.name for r in block]

        georeference_labels(settings.raw_dir / "sat", settings.raw_dir / "map")

        job = pipeline.new_job(req.name or f"{req.split} x{len(names)}")
        region = _finish_job(job, names, req.split)
        if quality:
            job.manifest["tile_quality"] = {
                n: q.to_dict() for n, q in quality.items() if n in set(names)
            }
            job.save()
        return {"job_id": job.job_id, "region": region, "tiles": names}

    return await asyncio.to_thread(_work)


@app.post("/api/jobs/upload")
async def upload_job(file: UploadFile = File(...), name: str = "") -> dict:
    """Create a job from a user-supplied georeferenced raster."""
    import rasterio

    settings = get_settings()
    suffix = Path(file.filename or "upload.tif").suffix.lower()
    if suffix not in (".tif", ".tiff"):
        raise HTTPException(400, "please upload a GeoTIFF (.tif/.tiff)")

    job = pipeline.new_job(name or Path(file.filename or "upload").stem)
    dest = job.dir / "mosaic.tif"
    with dest.open("wb") as fh:
        shutil.copyfileobj(file.file, fh)

    try:
        with rasterio.open(dest) as ds:
            if ds.crs is None:
                raise HTTPException(
                    400,
                    "that raster has no CRS. A georeferenced GeoTIFF is required "
                    "so results can be placed on a map and measured in metres.",
                )
            region = {
                "width": ds.width, "height": ds.height, "bands": ds.count,
                "crs": str(ds.crs), "n_tiles": 1, "truth": False,
                "source": "upload", "filename": file.filename,
            }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(400, f"could not read that raster: {exc}")

    region["bounds_wgs84"] = job.bounds_wgs84()
    job.manifest["region"] = region
    job.manifest["source"] = "upload"
    job.record("created", {"upload": file.filename}, region, 0.0)

    from gisagent.raster.preview import render_image_preview

    render_image_preview(job.image_path, job.preview_dir / "image.png")
    return {"job_id": job.job_id, "region": region}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str) -> dict:
    try:
        job = pipeline.get_job(job_id)
    except FileNotFoundError:
        raise HTTPException(404, f"no such job: {job_id}")
    status = job.status()
    status["manifest"] = job.manifest
    status["agent_state"] = registry.state(job_id)
    return status


@app.delete("/api/jobs/{job_id}")
def delete_job(job_id: str) -> dict:
    try:
        job = pipeline.get_job(job_id)
    except FileNotFoundError:
        raise HTTPException(404, f"no such job: {job_id}")
    shutil.rmtree(job.dir, ignore_errors=True)
    return {"deleted": job_id}


@app.get("/api/jobs/{job_id}/previews")
async def job_previews(job_id: str) -> dict:
    from gisagent.raster.preview import build_job_previews

    try:
        job = pipeline.get_job(job_id)
    except FileNotFoundError:
        raise HTTPException(404, f"no such job: {job_id}")
    previews = await asyncio.to_thread(build_job_previews, job)
    for meta in previews.values():
        meta["url"] = f"/api/jobs/{job_id}/preview/{meta['url_path']}"
    return {"job_id": job_id, "previews": previews,
            "bounds": job.bounds_wgs84() if job.image_path.exists() else None}


@app.get("/api/jobs/{job_id}/preview/{filename}")
def job_preview_file(job_id: str, filename: str):
    try:
        job = pipeline.get_job(job_id)
    except FileNotFoundError:
        raise HTTPException(404, f"no such job: {job_id}")
    path = (job.preview_dir / filename).resolve()
    if not path.is_file():
        raise HTTPException(404, "no such preview")
    return FileResponse(path, headers={"Cache-Control": "no-cache"})


@app.get("/api/jobs/{job_id}/chips")
async def job_chips(job_id: str) -> dict:
    from gisagent.raster.preview import render_chip_thumbnail

    try:
        job = pipeline.get_job(job_id)
    except FileNotFoundError:
        raise HTTPException(404, f"no such job: {job_id}")
    if not (job.chips_dir / "chips.json").exists():
        return {"job_id": job_id, "chips": []}

    def _work():
        out = []
        thumbs = job.preview_dir / "chips"
        for spec in job.chips():
            thumb = thumbs / f"{spec.chip_id}.png"
            if not thumb.exists() and spec.path:
                try:
                    render_chip_thumbnail(spec.path, thumb)
                except Exception:
                    continue
            out.append({
                "chip_id": spec.chip_id, "bounds": spec.bounds,
                "size": [spec.width, spec.height],
                "thumb": f"/api/jobs/{job_id}/preview/chips/{spec.chip_id}.png",
            })
        return out

    return {"job_id": job_id, "chips": await asyncio.to_thread(_work)}


@app.get("/api/jobs/{job_id}/preview/chips/{filename}")
def job_chip_thumb(job_id: str, filename: str):
    try:
        job = pipeline.get_job(job_id)
    except FileNotFoundError:
        raise HTTPException(404, f"no such job: {job_id}")
    path = (job.preview_dir / "chips" / filename).resolve()
    if not path.is_file():
        raise HTTPException(404, "no such thumbnail")
    return FileResponse(path)


@app.get("/api/jobs/{job_id}/roads.geojson")
def job_roads(job_id: str):
    try:
        job = pipeline.get_job(job_id)
    except FileNotFoundError:
        raise HTTPException(404, f"no such job: {job_id}")
    if not job.vector_path.exists():
        raise HTTPException(404, "not vectorized yet")
    return JSONResponse(json.loads(job.vector_path.read_text(encoding="utf-8")),
                        headers={"Cache-Control": "no-cache"})


# --------------------------------------------------------------------------- #
# human edits: the person finishing what the model started
# --------------------------------------------------------------------------- #

def _job_or_404(job_id: str):
    try:
        return pipeline.get_job(job_id)
    except FileNotFoundError:
        raise HTTPException(404, f"no such job: {job_id}")


@app.get("/api/jobs/{job_id}/network.geojson")
async def job_network(job_id: str):
    """The deliverable: machine centrelines with the person's edits applied."""
    job = _job_or_404(job_id)
    if not job.vector_path.exists() and not job.edits_path.exists():
        raise HTTPException(404, "not vectorized yet")
    fc = await asyncio.to_thread(job.network)
    return JSONResponse(fc, headers={"Cache-Control": "no-cache"})


@app.post("/api/jobs/{job_id}/edits")
async def apply_edit(job_id: str, req: EditRequest) -> dict:
    job = _job_or_404(job_id)
    log = job.edit_log()
    try:
        if req.op == "add":
            op = log.add_line(req.geometry, note=req.note)
        elif req.op == "delete":
            op = log.delete(req.target or {}, note=req.note)
        elif req.op == "replace":
            op = log.replace(req.target or {}, req.geometry, note=req.note)
        elif req.op == "dismiss":
            op = log.dismiss(req.geometry, note=req.note)
        else:
            raise ValueError(f"unknown op {req.op!r}")
    except (ValueError, TypeError) as exc:
        raise HTTPException(400, str(exc))
    return await _after_edit(job, op["op"], op)


@app.post("/api/jobs/{job_id}/edits/{action}")
async def undo_redo(job_id: str, action: str) -> dict:
    if action not in ("undo", "redo"):
        raise HTTPException(404, "expected undo or redo")
    job = _job_or_404(job_id)
    log = job.edit_log()
    op = log.undo() if action == "undo" else log.redo()
    if op is None:
        raise HTTPException(409, f"nothing to {action}")
    return await _after_edit(job, action, op)


async def _after_edit(job, action: str, op: dict) -> dict:
    fc = await asyncio.to_thread(job.rebuild_network)
    stats = fc["stats"]
    event = {"type": "edit", "action": action, "op": op.get("op"),
             "note": op.get("note", ""), "stats": stats}
    _log_event(job, event)
    await registry.publish(job.job_id, event)
    return {"action": action, "op": op, "stats": stats, "network": fc}


@app.get("/api/jobs/{job_id}/edits")
def list_edits(job_id: str) -> dict:
    job = _job_or_404(job_id)
    log = job.edit_log()
    state = log.state()
    return {"ops": log.ops[-200:], "n_ops": state.n_ops,
            "can_undo": state.n_ops > 0, "can_redo": state.can_redo,
            "n_human": len(state.human), "n_deleted": len(state.deleted)}


@app.get("/api/jobs/{job_id}/suggestions")
async def suggestions(job_id: str, threshold: float = 0.25, limit: int = 60) -> dict:
    job = _job_or_404(job_id)
    try:
        return await asyncio.to_thread(job.suggest, threshold, limit)
    except RuntimeError as exc:
        raise HTTPException(409, str(exc))


@app.get("/api/jobs/{job_id}/network/score")
async def network_score(job_id: str, tolerance_m: float = 5.0) -> dict:
    job = _job_or_404(job_id)
    if not job.has_truth():
        raise HTTPException(409, "no ground truth for this job")
    return await asyncio.to_thread(job.score_network, tolerance_m)


# --------------------------------------------------------------------------- #
# pipeline operations (through MCP, so the model stays in one process)
# --------------------------------------------------------------------------- #

@app.post("/api/jobs/{job_id}/pipeline")
async def run_pipeline(job_id: str, req: PipelineRequest) -> dict:
    try:
        pipeline.get_job(job_id)
    except FileNotFoundError:
        raise HTTPException(404, f"no such job: {job_id}")

    key = job_id
    registry.set_state(key, "running")

    async def step(tool: str, args: dict) -> dict:
        await registry.publish(key, {"type": "tool_call", "tool": tool, "args": args})
        res = await mcp_client.call(tool, args)
        from gisagent.agent.loop import summarise

        await registry.publish(key, {"type": "tool_result", "tool": tool,
                                     "result": res, "summary": summarise(tool, res)})
        if isinstance(res, dict) and res.get("ok") is False:
            raise HTTPException(500, f"{tool}: {res.get('error')}")
        return res

    try:
        out = {}
        out["tiling"] = await step("tile_region", {
            "job_id": job_id, "chip_size": req.chip_size, "overlap": req.overlap})
        out["segmentation"] = await step("segment_chips", {
            "job_id": job_id, "prompt": req.prompt,
            "threshold": req.seg_threshold, "upscale": req.upscale})
        out["stitch"] = await step("stitch_result", {
            "job_id": job_id, "threshold": req.mask_threshold})
        job = pipeline.get_job(job_id)
        if job.has_truth():
            out["metrics"] = await step("evaluate_result", {"job_id": job_id})
        out["vector"] = await step("vectorize_result", {"job_id": job_id})
        return out
    finally:
        registry.set_state(key, "idle")
        await registry.publish(key, {"type": "eof"})


@app.post("/api/jobs/{job_id}/refine")
async def refine(job_id: str, req: RefineRequest) -> dict:
    """Rework one area directly, without going through the LLM."""
    try:
        pipeline.get_job(job_id)
    except FileNotFoundError:
        raise HTTPException(404, f"no such job: {job_id}")
    args = {"job_id": job_id, "prompt": req.prompt, "threshold": req.threshold,
            "upscale": req.upscale, "note": req.note or req.area or "manual area"}
    if req.bbox_wgs84:
        args["bbox_wgs84"] = req.bbox_wgs84
    else:
        args["area"] = req.area or "whole"
    res = await mcp_client.call("refine_area", args)
    if isinstance(res, dict) and res.get("ok") is False:
        raise HTTPException(500, res.get("error", "refine failed"))
    return res


# --------------------------------------------------------------------------- #
# conversational agent
# --------------------------------------------------------------------------- #

@app.get("/api/jobs/{job_id}/conversation")
def get_conversation(job_id: str) -> dict:
    from gisagent.agent.loop import Conversation, load_plan
    from gisagent.checkpoints import Checkpoints

    try:
        job = pipeline.get_job(job_id)
    except FileNotFoundError:
        raise HTTPException(404, f"no such job: {job_id}")
    convo = Conversation(job.dir / "conversation.json")
    rewindable = {c["message_index"]: c["id"] for c in Checkpoints(job.dir).list()}
    events = []
    path = job.dir / "events.jsonl"
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    messages = convo.visible()
    for m in messages:
        if m["role"] == "user" and m["index"] in rewindable:
            m["checkpoint"] = rewindable[m["index"]]
    for e in events:
        if e.get("type") == "user":
            e["rewindable"] = e.get("index") in rewindable
    return {"job_id": job_id, "messages": messages, "events": events,
            "plan": load_plan(job.dir), "state": registry.state(job_id)}


@app.post("/api/jobs/{job_id}/rewind")
def rewind(job_id: str, req: RewindRequest) -> dict:
    """Undo a turn: restore the job's files and cut the conversation there."""
    from gisagent.agent.loop import Conversation
    from gisagent.checkpoints import Checkpoints

    job = _job_or_404(job_id)
    if registry.state(job_id) == "running":
        raise HTTPException(409, "the agent is working; cancel it first")
    cps = Checkpoints(job.dir)
    match = next((c for c in cps.list() if c["message_index"] == req.message_index), None)
    if match is None:
        raise HTTPException(404, "no checkpoint for that message")
    meta = cps.restore(match["id"])
    # the plan is part of the snapshot, so it went back with the files
    Conversation(job.dir / "conversation.json").truncate(meta["message_index"])
    # the replay log goes back to the same point as the files and the chat
    events = job.dir / "events.jsonl"
    if "events_offset" in meta and events.exists():
        with events.open("r+b") as fh:
            fh.truncate(meta["events_offset"])
    return {"rewound_to": match["id"], "label": meta["label"]}


def _log_event(job, payload: dict) -> None:
    """Append to the job's replay log, which rebuilds the agent panel on load."""
    with (job.dir / "events.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, default=str) + "\n")


@app.delete("/api/jobs/{job_id}/conversation")
def reset_conversation(job_id: str) -> dict:
    from gisagent.agent.loop import Conversation

    try:
        job = pipeline.get_job(job_id)
    except FileNotFoundError:
        raise HTTPException(404, f"no such job: {job_id}")
    Conversation(job.dir / "conversation.json").reset()
    (job.dir / "events.jsonl").write_text("", encoding="utf-8")
    (job.dir / "plan.json").unlink(missing_ok=True)
    return {"reset": job_id}


@app.post("/api/jobs/{job_id}/chat")
async def chat(job_id: str, req: ChatRequest) -> dict:
    from gisagent.agent.loop import Conversation, RoadAgent

    try:
        job = pipeline.get_job(job_id)
    except FileNotFoundError:
        raise HTTPException(404, f"no such job: {job_id}")
    if not get_settings().llm_ready():
        raise HTTPException(400, "no LLM API key configured (set OPENAI_API_KEY)")
    if registry.state(job_id) == "running":
        raise HTTPException(409, "the agent is already working on this job")
    if not req.message.strip():
        raise HTTPException(400, "message must not be empty")

    if req.mode not in ("work", "plan"):
        raise HTTPException(400, "mode must be 'work' or 'plan'")

    from gisagent.checkpoints import Checkpoints

    convo = Conversation(job.dir / "conversation.json")
    events_path = job.dir / "events.jsonl"
    checkpoints = Checkpoints(job.dir)

    async def _run() -> None:
        registry.set_state(job_id, "running")
        index = len(convo.messages)
        offset = events_path.stat().st_size if events_path.exists() else 0
        # snapshot before the turn, so it can be rewound files and all
        cp = await asyncio.to_thread(checkpoints.begin, req.message, index,
                                     events_offset=offset)
        _log_event(job, {"type": "user", "text": req.message, "index": index,
                         "mode": req.mode, "checkpoint": cp})
        await registry.publish(job_id, {"type": "checkpoint", "id": cp,
                                        "message_index": index})
        agent = RoadAgent(model=req.model, max_steps=req.max_steps)
        try:
            async for ev in agent.chat(convo, req.message, job_id=job_id,
                                       mode=req.mode):
                payload = ev.to_dict()
                _log_event(job, payload)
                await registry.publish(job_id, payload)
        except asyncio.CancelledError:
            _log_event(job, {"type": "error", "text": "stopped by the person"})
            await registry.publish(job_id, {"type": "error", "text": "stopped by the person"})
            raise
        except Exception as exc:
            _log_event(job, {"type": "error", "text": str(exc)})
            await registry.publish(job_id, {"type": "error", "text": str(exc)})
        finally:
            checkpoints.end()
            registry.set_state(job_id, "idle")
            await registry.publish(job_id, {"type": "eof"})

    registry.register(job_id, asyncio.create_task(_run()))
    return {"job_id": job_id, "state": "running"}


@app.post("/api/jobs/{job_id}/chat/cancel")
def cancel_chat(job_id: str) -> dict:
    return {"cancelled": registry.cancel(job_id)}


@app.websocket("/ws/jobs/{job_id}")
async def agent_ws(ws: WebSocket, job_id: str) -> None:
    await ws.accept()
    queue = registry.subscribe(job_id)
    try:
        await ws.send_json({"type": "connected", "state": registry.state(job_id)})
        while True:
            event = await queue.get()
            await ws.send_json(event)
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        registry.unsubscribe(job_id, queue)


if WEB_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")
