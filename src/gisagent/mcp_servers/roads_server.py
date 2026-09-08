"""MCP server exposing the road-annotation pipeline as agent tools.

Run it standalone with::

    python -m gisagent.mcp_servers.roads_server

Design note: the popular QGIS MCP servers on GitHub drive QGIS Desktop through a
plugin socket, which needs a human to click "Start Server" in the GUI and cannot
be containerised. This server instead calls qgis_process, the supported headless
entry point, so the same tools work in a container and behind a web backend.

The tools are deliberately fine-grained rather than one do-everything call. That
is what allows the agent to loop: segment, measure, change the prompt or the
threshold, and redo only the affected stage.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np

from mcp.server.mcpserver import MCPServer

from gisagent import pipeline
from gisagent.config import get_settings

mcp = MCPServer(
    name="gisagent-roads",
    instructions=(
        "Tools for annotating roads on satellite imagery. A typical loop is: "
        "create_region_job -> tile_region -> segment_chips -> stitch_result -> "
        "evaluate_result. If the score is poor, change the segmentation prompt "
        "or threshold and segment again, then re-evaluate. Finish with "
        "vectorize_result to produce road centrelines."
    ),
)

_segmenter = None


def _get_segmenter():
    """One model instance per server process; loading it costs seconds and VRAM."""
    global _segmenter
    if _segmenter is None:
        from gisagent.segment.sam3 import Sam3RoadSegmenter

        _segmenter = Sam3RoadSegmenter()
    return _segmenter


def _ok(**kw) -> dict:
    return {"ok": True, **kw}


def _err(message: str, **kw) -> dict:
    return {"ok": False, "error": message, **kw}


# --------------------------------------------------------------------------- #
# dataset discovery
# --------------------------------------------------------------------------- #

@mcp.tool(
    description=(
        "List satellite tiles available from the Massachusetts Roads dataset. "
        "Returns tile names plus their grid position, so spatially adjacent "
        "tiles can be chosen to form one contiguous region."
    )
)
def list_available_tiles(split: str = "train", limit: int = 20,
                         contiguous_block: int = 2) -> dict:
    """
    Args:
        split: which dataset split to list (train, valid or test).
        limit: maximum number of tiles to return.
        contiguous_block: if > 1, return an NxN block of adjacent tiles.
    """
    from gisagent.dataset.mass_roads import (
        list_split, find_contiguous_block, largest_connected_group,
    )

    try:
        refs = list_split(split)
    except Exception as exc:
        return _err(f"could not list split {split!r}: {exc}")

    block = None
    if contiguous_block and contiguous_block > 1:
        found = find_contiguous_block(refs, contiguous_block, contiguous_block)
        if found:
            block = [r.name for r in found]

    return _ok(
        split=split,
        total=len(refs),
        contiguous_block=block,
        largest_connected_region=len(largest_connected_group(refs)),
        tiles=[
            {"name": r.name, "key": [r.key_e, r.key_n],
             "bounds_26986": [round(v, 1) for v in r.bounds]}
            for r in refs[:limit]
        ],
    )


@mcp.tool(
    description=(
        "Download the given tiles and mosaic them into a new job. Ground-truth "
        "road masks are fetched too when available, which is what makes "
        "evaluate_result possible later."
    )
)
def create_region_job(tile_names: list[str], split: str = "train",
                      name: str = "") -> dict:
    """
    Args:
        tile_names: dataset tile names, e.g. ["22529485_15", "22679485_15"].
        split: the split those tiles belong to.
        name: optional human-readable label for the job.
    """
    from gisagent.dataset.mass_roads import TileRef, decode_name, download_tiles
    from gisagent.raster.georef import georeference_labels

    if not tile_names:
        return _err("tile_names must not be empty")

    settings = get_settings()
    raw = settings.raw_dir
    try:
        refs = []
        for n in tile_names:
            e, k = decode_name(n)
            refs.append(TileRef(name=n, split=split, key_e=e, key_n=k))
    except ValueError as exc:
        return _err(str(exc))

    results = download_tiles(refs, raw)
    failed = [r.ref.name for r in results if not r.ok]
    if failed:
        return _err(f"download failed for: {failed}",
                    detail=[r.errors for r in results if not r.ok])

    georeference_labels(raw / "sat", raw / "map")

    job = pipeline.new_job(name or f"{split}:{len(refs)} tiles")
    job.manifest["tiles"] = tile_names
    job.manifest["split"] = split
    sat = [raw / "sat" / f"{n}.tif" for n in tile_names]
    truth = [raw / "map" / f"{n}.tif" for n in tile_names]
    truth = [p for p in truth if p.exists()] or None

    try:
        region = job.build_region(sat, truth)
    except Exception as exc:
        return _err(f"mosaic failed: {exc}")

    return _ok(job_id=job.job_id, region=region)


# --------------------------------------------------------------------------- #
# job inspection
# --------------------------------------------------------------------------- #

@mcp.tool(description="Current stage, artefacts and latest metrics for a job.")
def get_job_status(job_id: str) -> dict:
    try:
        return _ok(**pipeline.get_job(job_id).status())
    except FileNotFoundError as exc:
        return _err(str(exc))


@mcp.tool(description="List all jobs, newest first.")
def list_jobs() -> dict:
    return _ok(jobs=pipeline.list_jobs())


# --------------------------------------------------------------------------- #
# tiling
# --------------------------------------------------------------------------- #

@mcp.tool(
    description=(
        "Cut the job's mosaic into overlapping chips for the segmentation model. "
        "Overlap matters: a model run per chip disagrees with itself at the "
        "seams, and the overlap is what lets those be blended away."
    )
)
def tile_region(job_id: str, chip_size: int = 1024, overlap: int = 128) -> dict:
    """
    Args:
        job_id: the job to tile.
        chip_size: chip edge length in pixels (1 px = 1 m in this dataset).
        overlap: overlap between neighbouring chips, in pixels.
    """
    try:
        job = pipeline.get_job(job_id)
        return _ok(**job.tile(chip_size=chip_size, overlap=overlap))
    except Exception as exc:
        return _err(str(exc))


@mcp.tool(
    description=(
        "Basic statistics for one chip (brightness, contrast, and how much of "
        "it is dark or bright). Useful for deciding whether a chip is mostly "
        "forest, water or built-up before choosing a prompt."
    )
)
def inspect_chip(job_id: str, chip_id: str) -> dict:
    import rasterio

    try:
        job = pipeline.get_job(job_id)
        spec = next((s for s in job.chips() if s.chip_id == chip_id), None)
        if spec is None:
            return _err(f"no such chip: {chip_id}")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with rasterio.open(spec.path) as ds:
                arr = ds.read().astype(np.float32)
        grey = arr.mean(axis=0)
        return _ok(
            chip_id=chip_id,
            size=[spec.width, spec.height],
            bounds=[round(v, 1) for v in spec.bounds],
            mean_brightness=round(float(grey.mean()), 2),
            std_brightness=round(float(grey.std()), 2),
            dark_fraction=round(float((grey < 60).mean()), 4),
            bright_fraction=round(float((grey > 190).mean()), 4),
        )
    except Exception as exc:
        return _err(str(exc))


# --------------------------------------------------------------------------- #
# segmentation
# --------------------------------------------------------------------------- #

@mcp.tool(
    description=(
        "Run SAM 3 road segmentation over the job's chips using a text prompt. "
        "Wording changes results: 'road' is the baseline, but phrases like "
        "'paved road', 'street', or 'highway' can behave differently. Raising "
        "upscale makes thin roads bigger for the model at the cost of GPU "
        "memory and time. Re-run with different settings to improve the score."
    )
)
def segment_chips(job_id: str, prompt: str = "road", threshold: float = 0.4,
                  upscale: int = 1, chip_ids: list[str] | None = None) -> dict:
    """
    Args:
        job_id: the job to segment.
        prompt: text concept to segment, e.g. "road".
        threshold: detection score cut-off for keeping an instance (0-1).
        upscale: integer upsampling factor applied before inference (1 = native).
        chip_ids: optionally restrict to specific chips.
    """
    try:
        job = pipeline.get_job(job_id)
        seg = _get_segmenter()
        result = job.segment(seg, prompt=prompt, threshold=threshold,
                             upscale=upscale, chip_ids=chip_ids)
        result.pop("per_chip", None)
        return _ok(**result)
    except Exception as exc:
        return _err(f"segmentation failed: {exc}")


@mcp.tool(
    description=(
        "Blend the per-chip confidence maps back into one georeferenced mask "
        "for the whole region, applying a probability threshold."
    )
)
def stitch_result(job_id: str, threshold: float = 0.5) -> dict:
    try:
        return _ok(**pipeline.get_job(job_id).stitch(threshold=threshold))
    except Exception as exc:
        return _err(str(exc))


# --------------------------------------------------------------------------- #
# evaluation
# --------------------------------------------------------------------------- #

@mcp.tool(
    description=(
        "Score the stitched mask against ground truth. Roads cover only a few "
        "percent of a tile, so IoU and F1 on the road class are the meaningful "
        "numbers; pixel accuracy is not. Relaxed scores allow a few pixels of "
        "positional slack, which is the convention for hand-drawn road labels."
    )
)
def evaluate_result(job_id: str, slack_px: int = 3) -> dict:
    try:
        return _ok(**pipeline.get_job(job_id).evaluate(slack_px=slack_px))
    except Exception as exc:
        return _err(str(exc))


@mcp.tool(
    description=(
        "Score the confidence map across many thresholds at once and report the "
        "best. Cheaper than re-segmenting: use this to pick an operating point "
        "before changing the prompt."
    )
)
def sweep_threshold(job_id: str, slack_px: int = 3) -> dict:
    from gisagent.evaluate.metrics import best_threshold

    try:
        rows = pipeline.get_job(job_id).sweep_threshold(slack_px=slack_px)
        trimmed = [
            {k: r[k] for k in ("threshold", "iou", "f1", "precision", "recall")}
            for r in rows
        ]
        return _ok(sweep=trimmed,
                   best_f1=best_threshold(rows, "f1"),
                   best_iou=best_threshold(rows, "iou"))
    except Exception as exc:
        return _err(str(exc))


# --------------------------------------------------------------------------- #
# vectorization
# --------------------------------------------------------------------------- #

@mcp.tool(
    description=(
        "Convert the road mask into vector centrelines (GeoJSON, WGS84). "
        "Cleans speckle, bridges gaps where tree canopy breaks a road, thins to "
        "a one-pixel skeleton, then traces polylines and simplifies them."
    )
)
def vectorize_result(job_id: str, min_object_px: int = 400, min_hole_px: int = 200,
                     close_radius: int = 2, simplify_tolerance_m: float = 2.0,
                     min_length_m: float = 25.0) -> dict:
    """
    Args:
        job_id: the job to vectorize.
        min_object_px: drop connected blobs smaller than this many pixels.
        min_hole_px: fill holes smaller than this many pixels.
        close_radius: morphological closing radius, in pixels, to bridge gaps.
        simplify_tolerance_m: Douglas-Peucker tolerance in metres.
        min_length_m: discard centrelines shorter than this.
    """
    try:
        job = pipeline.get_job(job_id)
        return _ok(**job.vectorize(
            min_object_px=min_object_px, min_hole_px=min_hole_px,
            close_radius=close_radius, simplify_tolerance_m=simplify_tolerance_m,
            min_length_m=min_length_m,
        ))
    except Exception as exc:
        return _err(str(exc))


# --------------------------------------------------------------------------- #
# targeted rework
# --------------------------------------------------------------------------- #

@mcp.tool(
    description=(
        "Re-run segmentation over ONE AREA of the region only, then rebuild the "
        "mask and vectors. Use this when a user says a specific part is wrong "
        "(e.g. 'the roads in the top-left are missing'). Far cheaper than "
        "re-running the whole region. Give either a named area or an explicit "
        "bounding box. Consider a different prompt or a higher upscale for the "
        "rework, since the defaults already failed there."
    )
)
def refine_area(job_id: str, area: str = "", bbox_wgs84: list[float] | None = None,
                prompt: str = "road network", threshold: float = 0.25,
                upscale: int = 2, note: str = "") -> dict:
    """
    Args:
        job_id: the job to correct.
        area: a named part of the region - one of top-left, top, top-right,
            left, center, right, bottom-left, bottom, bottom-right, or the
            whole region. Ignored when bbox_wgs84 is given.
        bbox_wgs84: explicit [west, south, east, north] in degrees.
        prompt: text prompt to use for the rework.
        threshold: detection score cut-off.
        upscale: upsampling factor; 2 often recovers roads the first pass missed.
        note: why this rework was requested, recorded in the job history.
    """
    try:
        job = pipeline.get_job(job_id)
    except FileNotFoundError as exc:
        return _err(str(exc))

    window = None
    if not bbox_wgs84:
        try:
            import rasterio

            with rasterio.open(job.image_path) as ds:
                W, H = ds.width, ds.height
        except Exception as exc:
            return _err(f"cannot read region: {exc}")
        window = _named_window(area or "whole", W, H)
        if window is None:
            return _err(
                f"unknown area {area!r}; use one of top-left, top, top-right, "
                "left, center, right, bottom-left, bottom, bottom-right, whole"
            )

    try:
        result = job.refine(_get_segmenter(), bbox_wgs84=bbox_wgs84,
                            window=window, prompt=prompt, threshold=threshold,
                            upscale=upscale, note=note or area)
        stitch = job.stitch(threshold=0.5)
        out = {**result, "stitch": stitch}
        if job.has_truth():
            out["metrics"] = job.evaluate()
        out["vector"] = job.vectorize()
        return _ok(**out)
    except Exception as exc:
        return _err(f"refine failed: {exc}")


def _named_window(area: str, width: int, height: int):
    """Map a spoken area name onto a pixel window, with a little overlap."""
    a = area.strip().lower().replace("_", "-").replace(" ", "-")
    if a in ("whole", "all", "everything", "full", "region"):
        return (0, 0, width, height)
    # thirds, so "top-left" is a generous quadrant-ish area rather than a sliver
    xs = {"left": (0, width * 0.45), "center": (width * 0.28, width * 0.72),
          "middle": (width * 0.28, width * 0.72), "right": (width * 0.55, width)}
    ys = {"top": (0, height * 0.45), "center": (height * 0.28, height * 0.72),
          "middle": (height * 0.28, height * 0.72), "bottom": (height * 0.55, height)}
    parts = a.split("-")
    vy = next((p for p in parts if p in ys), None)
    vx = next((p for p in parts if p in xs), None)
    if vy is None and vx is None:
        return None
    x0, x1 = xs.get(vx, (0, width))
    y0, y1 = ys.get(vy, (0, height))
    return (int(x0), int(y0), max(1, int(x1 - x0)), max(1, int(y1 - y0)))


@mcp.tool(
    description=(
        "Report the lowest-confidence roads found so far, with their locations. "
        "Use this to decide where a rework would help most, or to answer a user "
        "asking which parts of the annotation are least trustworthy."
    )
)
def low_confidence_roads(job_id: str, limit: int = 10) -> dict:
    try:
        job = pipeline.get_job(job_id)
    except FileNotFoundError as exc:
        return _err(str(exc))
    if not job.vector_path.exists():
        return _err("not vectorized yet")

    data = json.loads(job.vector_path.read_text(encoding="utf-8"))
    feats = []
    for f in data.get("features", []):
        props = f.get("properties") or {}
        coords = (f.get("geometry") or {}).get("coordinates") or []
        if not coords:
            continue
        mid = coords[len(coords) // 2]
        feats.append({
            "confidence_pct": props.get("confidence_pct", 0),
            "length_m": props.get("length_m", 0),
            "at": [round(mid[0], 5), round(mid[1], 5)],
        })
    feats.sort(key=lambda f: f["confidence_pct"])
    return _ok(total=len(feats), lowest=feats[:limit])


# --------------------------------------------------------------------------- #
# QGIS passthrough
# --------------------------------------------------------------------------- #

@mcp.tool(description="QGIS/GDAL/GRASS version string from the headless CLI.")
def qgis_version() -> dict:
    from gisagent.qgis.process import QgisProcess, QgisNotAvailable

    try:
        return _ok(version=QgisProcess().version())
    except QgisNotAvailable as exc:
        return _err(str(exc))


@mcp.tool(
    description=(
        "Search the installed QGIS processing algorithms by keyword, e.g. "
        "'polygonize', 'simplify', 'buffer'. Returns algorithm ids usable with "
        "run_qgis_algorithm."
    )
)
def list_qgis_algorithms(filter: str = "", limit: int = 40) -> dict:
    from gisagent.qgis.process import QgisProcess, QgisNotAvailable

    try:
        algs = QgisProcess().list_algorithms()
    except QgisNotAvailable as exc:
        return _err(str(exc))
    needle = filter.lower()
    hits = {k: v for k, v in algs.items()
            if not needle or needle in k.lower() or needle in v.lower()}
    return _ok(total=len(algs), matched=len(hits),
               algorithms=dict(list(hits.items())[:limit]))


@mcp.tool(
    description=(
        "Run any QGIS processing algorithm headlessly. Parameters are given as "
        "a JSON object, e.g. {\"INPUT\": \"in.tif\", \"OUTPUT\": \"out.gpkg\"}."
    )
)
def run_qgis_algorithm(algorithm: str, params_json: str = "{}",
                       timeout_s: float = 900) -> dict:
    from gisagent.qgis.process import QgisProcess, QgisNotAvailable, QgisAlgorithmError

    try:
        params = json.loads(params_json) if params_json.strip() else {}
    except json.JSONDecodeError as exc:
        return _err(f"params_json is not valid JSON: {exc}")
    if not isinstance(params, dict):
        return _err("params_json must decode to a JSON object")

    try:
        res = QgisProcess().run(algorithm, params, timeout=timeout_s)
        return _ok(algorithm=algorithm, outputs=res.outputs,
                   duration_s=round(res.duration_s, 2))
    except (QgisNotAvailable, QgisAlgorithmError) as exc:
        return _err(str(exc))
    except Exception as exc:
        return _err(f"unexpected failure: {exc}")


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
