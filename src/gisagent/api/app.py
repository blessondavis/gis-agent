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


class PipelineRequest(BaseModel):
    chip_size: int = 1024
    overlap: int = 128
    prompt: str = "road network"
    seg_threshold: float = 0.3
    upscale: int = 1
    mask_threshold: float = 0.5


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
        TileRef, decode_name, download_tiles, find_best_block,
        find_contiguous_block, list_split,
    )
    from gisagent.raster.georef import georeference_labels

    settings = get_settings()

    def _work() -> dict:
        names = req.tile_names
        quality = {}
        if not names:
            refs_all = list_split(req.split)
            if req.screen and req.block > 1:
                block, quality = find_best_block(
                    refs_all, req.block, max_blank=0.12, min_road=0.006,
                    search_limit=60,
                )
            else:
                block = find_contiguous_block(refs_all, req.block, req.block)
            if not block:
                raise HTTPException(400, "no suitable block found")
            names = [r.name for r in block]

        refs = []
        for n in names:
            e, k = decode_name(n)
            refs.append(TileRef(name=n, split=req.split, key_e=e, key_n=k))

        results = download_tiles(refs, settings.raw_dir)
        failed = [r.ref.name for r in results if not r.ok]
        if failed:
            raise HTTPException(502, f"tile download failed: {failed}")
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

def _job_context(job) -> str:
    s = job.status()
    lines = [
        f"job_id: {s['job_id']}",
        f"stage: {s['stage']}",
        f"region: {s['region'].get('width')}x{s['region'].get('height')} px, "
        f"{s['region'].get('n_tiles')} tile(s), CRS {s['region'].get('crs')}",
        f"ground truth available: {bool(s['region'].get('truth'))}",
    ]
    if s.get("metrics"):
        m = s["metrics"]
        lines.append(f"latest metrics: IoU {m['iou']}, F1 {m['f1']}, "
                     f"relaxed F1 {m['relaxed_f1']}")
    if s.get("vector_stats"):
        v = s["vector_stats"]
        lines.append(f"vectors: {v['n_features']} centrelines, "
                     f"{v['total_length_km']} km")
    seg = job.manifest.get("segmentation")
    if seg:
        lines.append(f"last segmentation: prompt '{seg.get('prompt')}', "
                     f"threshold {seg.get('threshold')}, upscale {seg.get('upscale')}")
    refs = job.manifest.get("refinements") or []
    if refs:
        lines.append(f"areas already reworked: "
                     f"{', '.join(r.get('note', '?') for r in refs[-4:])}")
    return "\n".join(lines)


@app.get("/api/jobs/{job_id}/conversation")
def get_conversation(job_id: str) -> dict:
    from gisagent.agent.loop import Conversation

    try:
        job = pipeline.get_job(job_id)
    except FileNotFoundError:
        raise HTTPException(404, f"no such job: {job_id}")
    convo = Conversation(job.dir / "conversation.json")
    events = []
    path = job.dir / "events.jsonl"
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return {"job_id": job_id, "messages": convo.visible(), "events": events,
            "state": registry.state(job_id)}


@app.delete("/api/jobs/{job_id}/conversation")
def reset_conversation(job_id: str) -> dict:
    from gisagent.agent.loop import Conversation

    try:
        job = pipeline.get_job(job_id)
    except FileNotFoundError:
        raise HTTPException(404, f"no such job: {job_id}")
    Conversation(job.dir / "conversation.json").reset()
    (job.dir / "events.jsonl").write_text("", encoding="utf-8")
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

    convo = Conversation(job.dir / "conversation.json")
    events_path = job.dir / "events.jsonl"

    async def _run() -> None:
        registry.set_state(job_id, "running")
        agent = RoadAgent(model=req.model, max_steps=req.max_steps)
        try:
            async for ev in agent.chat(convo, req.message,
                                       context=_job_context(job)):
                payload = ev.to_dict()
                with events_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(payload, default=str) + "\n")
                await registry.publish(job_id, payload)
        except asyncio.CancelledError:
            await registry.publish(job_id, {"type": "error", "text": "cancelled"})
            raise
        except Exception as exc:
            await registry.publish(job_id, {"type": "error", "text": str(exc)})
        finally:
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
