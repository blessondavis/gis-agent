"""Turn a binary road mask into road vectors.

Roads are linear features, so the useful deliverable is a *centreline* network,
not a blob outline. The chain is:

    mask -> morphological cleanup -> 1-px skeleton -> line vectors -> simplify

Skeletonisation is done in Python (fast, deterministic, no GRASS region quirks)
and the skeleton-to-line conversion is handed to GRASS r.to.vect through QGIS,
which already solves the fiddly part: walking a pixel skeleton into connected
polylines with proper junctions. A pure-Python tracer stands in when QGIS is
unavailable so the module still works in a bare container.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import Affine


@dataclass
class VectorizeStats:
    """What the vectoriser did, in units a human can sanity-check."""

    mask_pixels: int = 0
    mask_fraction: float = 0.0
    removed_small_objects: int = 0
    filled_holes: int = 0
    skeleton_pixels: int = 0
    n_features: int = 0
    total_length_m: float = 0.0
    mean_confidence: float = 0.0
    low_confidence_features: int = 0
    crs: str = ""

    def to_dict(self) -> dict:
        return {
            "mask_pixels": self.mask_pixels,
            "mask_fraction": round(self.mask_fraction, 6),
            "removed_small_objects": self.removed_small_objects,
            "filled_holes": self.filled_holes,
            "skeleton_pixels": self.skeleton_pixels,
            "n_features": self.n_features,
            "total_length_m": round(self.total_length_m, 1),
            "total_length_km": round(self.total_length_m / 1000.0, 3),
            "mean_confidence": round(self.mean_confidence, 4),
            "low_confidence_features": self.low_confidence_features,
            "crs": self.crs,
        }


# --------------------------------------------------------------------------- #
# raster-domain cleanup
# --------------------------------------------------------------------------- #

def clean_mask(
    mask: np.ndarray,
    *,
    min_object_px: int = 400,
    min_hole_px: int = 200,
    close_radius: int = 2,
) -> tuple[np.ndarray, dict]:
    """Remove speckle, bridge small gaps, and fill pinholes in a binary mask.

    Segmentation output on 1 m imagery is typically broken wherever tree canopy
    crosses a road, so a small closing before skeletonising avoids shattering
    one road into a dozen fragments.
    """
    from skimage.morphology import (
        closing, disk, remove_small_holes, remove_small_objects,
    )
    from skimage.measure import label

    binary = np.asarray(mask).astype(bool)
    stats = {"removed_small_objects": 0, "filled_holes": 0}

    if close_radius > 0:
        binary = closing(binary, disk(close_radius))

    # skimage >= 0.26 takes max_size, and drops features <= that value, so the
    # exclusive "smaller than N" threshold is expressed as max_size = N - 1.
    if min_hole_px > 0:
        before = binary.sum()
        binary = remove_small_holes(binary, max_size=max(min_hole_px - 1, 0))
        stats["filled_holes"] = int(binary.sum() - before)

    if min_object_px > 0:
        n_before = label(binary, connectivity=2).max()
        binary = remove_small_objects(
            binary, max_size=max(min_object_px - 1, 0), connectivity=2
        )
        n_after = label(binary, connectivity=2).max()
        stats["removed_small_objects"] = int(n_before - n_after)

    return binary, stats


def skeletonize_mask(mask: np.ndarray) -> np.ndarray:
    """Reduce a road mask to a 1-pixel-wide centreline skeleton."""
    from skimage.morphology import skeletonize

    return skeletonize(np.asarray(mask).astype(bool))


# --------------------------------------------------------------------------- #
# skeleton -> polylines
# --------------------------------------------------------------------------- #

_NEIGHBOURS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def trace_skeleton(skeleton: np.ndarray, transform: Affine) -> list[list[tuple[float, float]]]:
    """Walk a pixel skeleton into polylines, splitting at junctions.

    Endpoints (1 neighbour) and junctions (3+) become segment boundaries, so the
    result is a proper edge set rather than one tangled path.
    """
    skel = np.asarray(skeleton).astype(bool)
    H, W = skel.shape
    pts = {(int(r), int(c)) for r, c in zip(*np.nonzero(skel))}

    def nbrs(p):
        r, c = p
        out = []
        for dr, dc in _NEIGHBOURS:
            q = (r + dr, c + dc)
            if 0 <= q[0] < H and 0 <= q[1] < W and q in pts:
                out.append(q)
        return out

    degree = {p: len(nbrs(p)) for p in pts}
    nodes = {p for p, d in degree.items() if d != 2}          # ends + junctions
    visited_edges: set[frozenset] = set()
    lines: list[list[tuple[int, int]]] = []

    def walk(start, first):
        path = [start, first]
        prev, cur = start, first
        while degree.get(cur, 0) == 2:
            nxt = [q for q in nbrs(cur) if q != prev]
            if not nxt:
                break
            prev, cur = cur, nxt[0]
            path.append(cur)
        return path

    for node in nodes:
        for nb in nbrs(node):
            key = frozenset((node, nb))
            if key in visited_edges:
                continue
            path = walk(node, nb)
            for a, b in zip(path, path[1:]):
                visited_edges.add(frozenset((a, b)))
            if len(path) > 1:
                lines.append(path)

    # closed loops have no degree != 2 node; pick an arbitrary start
    remaining = {p for p in pts if degree.get(p) == 2}
    for p in list(remaining):
        if any(frozenset((p, q)) in visited_edges for q in nbrs(p)):
            continue
        path = walk(p, nbrs(p)[0]) if nbrs(p) else []
        for a, b in zip(path, path[1:]):
            visited_edges.add(frozenset((a, b)))
        if len(path) > 1:
            lines.append(path)

    # pixel centres -> CRS coordinates
    out: list[list[tuple[float, float]]] = []
    for path in lines:
        coords = [transform @ (c + 0.5, r + 0.5) for r, c in path]
        out.append([(float(x), float(y)) for x, y in coords])
    return out


def _douglas_peucker(coords, tol):
    from shapely.geometry import LineString

    if tol <= 0 or len(coords) < 3:
        return coords
    simplified = LineString(coords).simplify(tol, preserve_topology=False)
    return list(simplified.coords)


# --------------------------------------------------------------------------- #
# top-level entry point
# --------------------------------------------------------------------------- #

def _sample_confidence(conf: np.ndarray, inv_transform, coords) -> float:
    """Mean model confidence along a centreline, for per-road reporting."""
    if conf is None or not len(coords):
        return 0.0
    h, w = conf.shape
    vals = []
    for x, y in coords:
        col, row = inv_transform @ (x, y)
        c, r = int(col), int(row)
        if 0 <= r < h and 0 <= c < w:
            vals.append(float(conf[r, c]))
    return float(np.mean(vals)) if vals else 0.0


def vectorize_roads(
    mask_path: Path | str,
    out_path: Path | str,
    *,
    min_object_px: int = 400,
    min_hole_px: int = 200,
    close_radius: int = 2,
    simplify_tolerance_m: float = 2.0,
    min_length_m: float = 25.0,
    driver: str = "GeoJSON",
    to_wgs84: bool = False,
    confidence_path: Path | str | None = None,
) -> tuple[Path, VectorizeStats]:
    """Full mask -> centreline pipeline. Returns (output_path, stats).

    When a confidence raster is supplied, each output centreline carries the
    mean model confidence along its length, so the UI can show how sure the
    model was about that particular road rather than only a global score.
    """
    import geopandas as gpd
    from shapely.geometry import LineString

    mask_path = Path(mask_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(mask_path) as ds:
        raw = ds.read(1)
        transform = ds.transform
        crs = ds.crs

    conf = None
    if confidence_path is not None and Path(confidence_path).exists():
        with rasterio.open(confidence_path) as ds:
            conf = ds.read(1).astype(np.float32)
        if conf.shape != raw.shape:
            conf = None
    inv_transform = ~transform

    binary = raw > (127 if raw.dtype == np.uint8 else 0.5)
    stats = VectorizeStats(
        mask_pixels=int(binary.sum()),
        mask_fraction=float(binary.mean()),
        crs=str(crs),
    )

    cleaned, clean_stats = clean_mask(
        binary,
        min_object_px=min_object_px,
        min_hole_px=min_hole_px,
        close_radius=close_radius,
    )
    stats.removed_small_objects = clean_stats["removed_small_objects"]
    stats.filled_holes = clean_stats["filled_holes"]

    skeleton = skeletonize_mask(cleaned)
    stats.skeleton_pixels = int(skeleton.sum())

    paths = trace_skeleton(skeleton, transform)

    geoms, lengths, confs = [], [], []
    for coords in paths:
        coords = _douglas_peucker(coords, simplify_tolerance_m)
        if len(coords) < 2:
            continue
        line = LineString(coords)
        if line.length < min_length_m:
            continue
        geoms.append(line)
        lengths.append(line.length)
        confs.append(_sample_confidence(conf, inv_transform, coords))

    gdf = gpd.GeoDataFrame(
        {
            "length_m": [round(v, 2) for v in lengths],
            "confidence": [round(v, 4) for v in confs],
            "confidence_pct": [round(100 * v, 1) for v in confs],
        },
        geometry=geoms,
        crs=crs,
    )
    stats.n_features = len(gdf)
    stats.total_length_m = float(sum(lengths))
    if confs:
        stats.mean_confidence = float(np.mean(confs))
        stats.low_confidence_features = int(sum(1 for c in confs if c < 0.5))

    if to_wgs84 and len(gdf):
        gdf = gdf.to_crs("EPSG:4326")
    elif to_wgs84:
        gdf = gdf.set_crs(crs, allow_override=True).to_crs("EPSG:4326")

    if len(gdf) == 0:
        # geopandas refuses to write an empty frame with no geometry type
        empty = {
            "type": "FeatureCollection",
            "features": [],
            "crs": {"type": "name", "properties": {"name": str(gdf.crs)}},
        }
        out_path.write_text(json.dumps(empty), encoding="utf-8")
    else:
        gdf.to_file(out_path, driver=driver)

    return out_path, stats
