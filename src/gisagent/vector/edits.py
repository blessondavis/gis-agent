"""Human edits, layered over the machine-extracted road network.

The model gets most of a network; a person finishes it. The obvious way to
support that -- let the person edit ``roads.geojson`` in place -- does not
survive contact with the agent: every ``vectorize_result`` or ``refine_area``
rewrites that file, and the person's work would silently vanish.

So edits are kept as their own layer, an append-only log of operations in
``edits.json``, and merged over the machine output whenever the network is
read. The machine layer can be regenerated any number of times; the human layer
is replayed on top of whatever it produces.

Machine features have no stable identity across re-vectorisation, so a deletion
is recorded as the *geometry* that was deleted, not an id. On merge, any machine
line lying mostly along a deleted geometry is suppressed -- which still does the
right thing after the agent re-traces the same road slightly differently.

Human work wins conflicts: a machine line that runs along a human-drawn one is
dropped as a duplicate. A human line that ends on another road is snapped to
it, and that road is split there, so the junction exists in the graph and
topology scoring sees a connected network rather than a dead end.
"""

from __future__ import annotations

import json
import math
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

# An endpoint this close to another road is joined to it. Larger than the
# topology snap (2.5 m) because a person clicking at map zoom is less precise
# than a skeleton, but well under a road width.
SNAP_M = 3.0

# A machine line is "along" a deleted or human geometry when this share of its
# length lies within this distance of it. The buffer is about half a road
# width; the share keeps a crossing street (a few metres inside the buffer)
# from being deleted along with the street it crosses.
ALONG_BUFFER_M = 4.0
DELETE_SHARE = 0.6
SUPERSEDE_SHARE = 0.7

# Split points closer than this to an existing vertex end reuse that end.
_END_EPS_M = 0.5

OPS = ("add", "delete", "replace", "dismiss")


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _new_id() -> str:
    return "h-" + uuid.uuid4().hex[:8]


def _valid_line(coords) -> list[list[float]]:
    """Validate a [[lng, lat], ...] polyline from a client."""
    if not isinstance(coords, (list, tuple)) or len(coords) < 2:
        raise ValueError("a road needs at least two vertices")
    out = []
    for pt in coords:
        if not isinstance(pt, (list, tuple)) or len(pt) < 2:
            raise ValueError(f"bad vertex {pt!r}")
        lng, lat = float(pt[0]), float(pt[1])
        if not (-180 <= lng <= 180 and -90 <= lat <= 90) or math.isnan(lng + lat):
            raise ValueError(f"vertex out of range: {pt!r}")
        if not out or (lng, lat) != tuple(out[-1]):
            out.append([lng, lat])
    if len(out) < 2:
        raise ValueError("a road needs at least two distinct vertices")
    return out


# --------------------------------------------------------------------------- #
# the log
# --------------------------------------------------------------------------- #

@dataclass
class EditOp:
    op: str
    id: str = ""                       # the human feature this op creates
    geometry: list = field(default_factory=list)   # WGS84 [[lng, lat], ...]
    target: dict = field(default_factory=dict)     # {source, id, geometry}
    note: str = ""
    at: str = field(default_factory=_utc)


class EditLog:
    """Append-only edit operations for one job, with undo and redo."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.ops: list[dict] = []
        self.redo_stack: list[dict] = []
        if self.path.exists():
            data = json.loads(self.path.read_text(encoding="utf-8") or "{}")
            self.ops = list(data.get("ops", []))
            self.redo_stack = list(data.get("redo", []))

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"version": 1, "ops": self.ops,
                                   "redo": self.redo_stack}, indent=1),
                       encoding="utf-8")
        tmp.replace(self.path)

    def _push(self, op: EditOp) -> dict:
        d = asdict(op)
        self.ops.append(d)
        self.redo_stack.clear()        # a new edit forks history
        self.save()
        return d

    # -- operations --------------------------------------------------------- #

    def add_line(self, coords, note: str = "") -> dict:
        return self._push(EditOp(op="add", id=_new_id(),
                                 geometry=_valid_line(coords), note=note))

    def delete(self, target: dict, note: str = "") -> dict:
        return self._push(EditOp(op="delete", target=self._target(target), note=note))

    def replace(self, target: dict, coords, note: str = "") -> dict:
        return self._push(EditOp(op="replace", id=_new_id(),
                                 geometry=_valid_line(coords),
                                 target=self._target(target), note=note))

    def dismiss(self, geometry, note: str = "") -> dict:
        """Reject a suggestion so it is not offered again."""
        return self._push(EditOp(op="dismiss", geometry=_valid_line(geometry),
                                 note=note))

    def undo(self) -> dict | None:
        if not self.ops:
            return None
        op = self.ops.pop()
        self.redo_stack.append(op)
        self.save()
        return op

    def redo(self) -> dict | None:
        if not self.redo_stack:
            return None
        op = self.redo_stack.pop()
        self.ops.append(op)
        self.save()
        return op

    @staticmethod
    def _target(target: dict) -> dict:
        src = target.get("source")
        if src not in ("model", "human"):
            raise ValueError("target.source must be 'model' or 'human'")
        t = {"source": src, "id": str(target.get("id", ""))}
        if src == "model":
            # model ids are not stable across re-vectorisation; the geometry is
            t["geometry"] = _valid_line(target.get("geometry"))
        elif not t["id"]:
            raise ValueError("a human target needs its id")
        return t

    # -- replay ------------------------------------------------------------- #

    def state(self) -> "EditState":
        human: dict[str, dict] = {}
        deleted: list[list] = []
        dismissed: list[list] = []
        for op in self.ops:
            kind = op.get("op")
            if kind in ("delete", "replace"):
                t = op.get("target") or {}
                if t.get("source") == "human":
                    human.pop(t.get("id"), None)
                elif t.get("geometry"):
                    deleted.append(t["geometry"])
            if kind in ("add", "replace"):
                human[op["id"]] = {"geometry": op["geometry"],
                                   "note": op.get("note", ""), "at": op.get("at")}
            if kind == "dismiss":
                dismissed.append(op["geometry"])
        return EditState(human=human, deleted=deleted, dismissed=dismissed,
                         n_ops=len(self.ops), can_redo=bool(self.redo_stack))


@dataclass
class EditState:
    human: dict[str, dict]
    deleted: list[list]
    dismissed: list[list]
    n_ops: int = 0
    can_redo: bool = False


# --------------------------------------------------------------------------- #
# merge
# --------------------------------------------------------------------------- #

def _metric_crs(crs_hint: str | None, lines_wgs84: list[list]):
    """A metre-based CRS: the job's own if it is projected, else local UTM."""
    from pyproj import CRS

    if crs_hint:
        try:
            crs = CRS.from_user_input(crs_hint)
            if crs.is_projected:
                return crs
        except Exception:
            pass
    lng = lat = 0.0
    n = 0
    for line in lines_wgs84:
        for x, y in line:
            lng += x
            lat += y
            n += 1
    lng, lat = (lng / n, lat / n) if n else (0.0, 0.0)
    zone = int((lng + 180) // 6) + 1
    return CRS.from_epsg((32600 if lat >= 0 else 32700) + zone)


class _Along:
    """Answers "what share of this line runs along these geometries?".

    Buffers once and prepares the result, so testing thousands of machine
    lines against a handful of edits costs one cheap bbox-level check each,
    and a real intersection only for the few that are actually nearby.
    """

    def __init__(self, geoms: list, buffer_m: float) -> None:
        from shapely.ops import unary_union
        from shapely.prepared import prep

        self.zone = unary_union([g.buffer(buffer_m) for g in geoms]) if geoms else None
        self._prep = prep(self.zone) if self.zone is not None else None

    def share(self, line) -> float:
        if self.zone is None or line.length == 0 or not self._prep.intersects(line):
            return 0.0
        return line.intersection(self.zone).length / line.length


def _split_at(line, cuts: list[float]):
    """Split a LineString at distances along it."""
    from shapely.ops import substring

    cuts = sorted(d for d in cuts if _END_EPS_M < d < line.length - _END_EPS_M)
    if not cuts:
        return [line]
    pieces, prev = [], 0.0
    for d in cuts + [line.length]:
        if d - prev > _END_EPS_M:
            pieces.append(substring(line, prev, d))
        prev = d
    return [p for p in pieces if p.length > 0]


@dataclass
class NetworkStats:
    n_features: int = 0
    n_model: int = 0
    n_human: int = 0
    model_length_m: float = 0.0
    human_length_m: float = 0.0
    total_length_m: float = 0.0
    human_share: float = 0.0
    n_deleted: int = 0
    n_superseded: int = 0
    n_snapped_ends: int = 0
    n_ops: int = 0
    can_undo: bool = False
    can_redo: bool = False

    def to_dict(self) -> dict:
        d = asdict(self)
        for k in ("model_length_m", "human_length_m", "total_length_m"):
            d[k] = round(d[k], 1)
        d["human_share"] = round(d["human_share"], 4)
        d["total_length_km"] = round(self.total_length_m / 1000, 3)
        return d


def merge_network(machine: dict | None, state: EditState, *,
                  crs_hint: str | None = None, snap_m: float = SNAP_M
                  ) -> tuple[dict, NetworkStats]:
    """Machine centrelines + human edits -> the deliverable network (WGS84)."""
    from pyproj import Transformer
    from shapely.geometry import LineString, Point, mapping
    from shapely.ops import nearest_points, transform
    from shapely.strtree import STRtree

    machine_feats = []
    for i, f in enumerate((machine or {}).get("features", [])):
        g = f.get("geometry") or {}
        parts = ([g.get("coordinates")] if g.get("type") == "LineString"
                 else g.get("coordinates") or [] if g.get("type") == "MultiLineString"
                 else [])
        for j, coords in enumerate(parts):
            if coords and len(coords) >= 2:
                machine_feats.append((f"m-{i}" + (f".{j}" if j else ""),
                                      coords, f.get("properties") or {}))

    all_wgs = ([c for _, c, _ in machine_feats]
               + [h["geometry"] for h in state.human.values()] + state.deleted)
    metric = _metric_crs(crs_hint, all_wgs)
    fwd = Transformer.from_crs("EPSG:4326", metric, always_xy=True).transform
    inv = Transformer.from_crs(metric, "EPSG:4326", always_xy=True).transform

    def proj(coords):
        return transform(fwd, LineString(coords))

    stats = NetworkStats(n_ops=state.n_ops, can_undo=state.n_ops > 0,
                         can_redo=state.can_redo)

    # records: [kind, id, metric_line, props]
    model = [["model", fid, proj(c), p] for fid, c, p in machine_feats]
    human = [["human", hid, proj(h["geometry"]),
              {"note": h.get("note", ""), "edited_at": h.get("at")}]
             for hid, h in state.human.items()]

    # 1. human endpoints snap onto the network; roads they land on get split
    targets = model + human
    tree = STRtree([r[2] for r in targets]) if targets else None
    cuts: dict[int, list[float]] = {}
    for rec in human:
        line = rec[2]
        coords = list(line.coords)
        for end in (0, -1):
            pt = Point(coords[end])
            if tree is None:
                continue
            best, best_d = None, snap_m
            for k in tree.query(pt.buffer(snap_m)):
                other = targets[int(k)]
                if other is rec:
                    continue
                d = other[2].distance(pt)
                if d <= best_d:
                    best, best_d = int(k), d
            if best is None:
                continue
            other_line = targets[best][2]
            on = nearest_points(other_line, pt)[0]
            # prefer the other road's own endpoint if the snap lands near it
            for cand in (Point(other_line.coords[0]), Point(other_line.coords[-1])):
                if cand.distance(on) <= snap_m:
                    on = cand
                    break
            else:
                cuts.setdefault(best, []).append(other_line.project(on))
            coords[end] = (on.x, on.y)
            stats.n_snapped_ends += 1
        rec[2] = LineString(coords)

    for idx, dists in cuts.items():
        rec = targets[idx]
        pieces = _split_at(rec[2], dists)
        rec[2] = pieces[0]
        for n, piece in enumerate(pieces[1:], start=1):
            extra = [rec[0], f"{rec[1]}:{n}", piece, rec[3]]
            (model if rec[0] == "model" else human).append(extra)

    # 2. deletions and human redraws suppress the machine lines they run along
    deleted = _Along([proj(c) for c in state.deleted], ALONG_BUFFER_M)
    redrawn = _Along([r[2] for r in human], ALONG_BUFFER_M)
    kept = []
    for rec in model:
        line = rec[2]
        if deleted.share(line) >= DELETE_SHARE:
            stats.n_deleted += 1
            continue
        if redrawn.share(line) >= SUPERSEDE_SHARE:
            stats.n_superseded += 1
            continue
        kept.append(rec)

    # 3. assemble, back in WGS84
    features = []
    for kind, fid, line, props in kept + human:
        out_props = {"id": fid, "source": kind, "length_m": round(line.length, 2)}
        if kind == "model":
            out_props["confidence"] = props.get("confidence")
            out_props["confidence_pct"] = props.get("confidence_pct")
            stats.model_length_m += line.length
            stats.n_model += 1
        else:
            out_props.update({k: v for k, v in props.items() if v})
            stats.human_length_m += line.length
            stats.n_human += 1
        geom = mapping(transform(inv, line))
        features.append({"type": "Feature", "properties": out_props,
                         "geometry": {"type": "LineString",
                                      "coordinates": [list(map(float, c))
                                                      for c in geom["coordinates"]]}})

    stats.n_features = len(features)
    stats.total_length_m = stats.model_length_m + stats.human_length_m
    stats.human_share = (stats.human_length_m / stats.total_length_m
                         if stats.total_length_m else 0.0)
    fc = {"type": "FeatureCollection", "features": features,
          "stats": stats.to_dict()}
    return fc, stats


# --------------------------------------------------------------------------- #
# suggestions: help the person find the last few roads
# --------------------------------------------------------------------------- #

def _dangle_pairs(lines: list, max_gap_m: float, join_m: float = 2.5):
    """Pairs of dead ends within ``max_gap_m`` of each other.

    An end is only a dead end if it touches no other line -- neither another
    line's end (a junction) nor its interior (a T-junction).
    """
    from shapely.geometry import Point
    from shapely.strtree import STRtree

    if not lines:
        return []
    line_tree = STRtree(lines)
    dangles = []
    for i, ln in enumerate(lines):
        for p in (Point(ln.coords[0]), Point(ln.coords[-1])):
            touching = [int(k) for k in line_tree.query(p.buffer(join_m))
                        if int(k) != i and lines[int(k)].distance(p) <= join_m]
            if not touching:
                dangles.append((i, p))

    pts = [p for _, p in dangles]
    pt_tree = STRtree(pts) if pts else None
    pairs = []
    for a_idx, (line_a, a) in enumerate(dangles):
        for k in pt_tree.query(a.buffer(max_gap_m)):
            b_idx = int(k)
            if b_idx <= a_idx:
                continue
            line_b, b = dangles[b_idx]
            # the two ends of one short stub are not a gap
            if line_a == line_b:
                continue
            if not join_m < a.distance(b) <= max_gap_m:
                continue
            # A street that stops short carries on roughly straight. Joining
            # two dead ends that point elsewhere is how a gap-filler invents
            # roads across courtyards.
            align = max(_continuation(lines[line_a], a, b),
                        _continuation(lines[line_b], b, a))
            if align >= GAP_MIN_ALIGN:
                pairs.append((a, b, align))
    return pairs


# cos of the largest angle between a stub's heading and the connector: 40 deg
GAP_MIN_ALIGN = math.cos(math.radians(40))

# A suggested road's end this close to the network counts as joined to it.
CONNECT_M = 10.0


def _dist_at(dist, inv_tf, xy) -> float:
    if dist is None:
        return math.inf
    c, r = inv_tf * (xy[0], xy[1])
    r = min(max(int(r), 0), dist.shape[0] - 1)
    c = min(max(int(c), 0), dist.shape[1] - 1)
    return float(dist[r, c])


def _continuation(line, end, towards) -> float:
    """Cosine between the line's heading at ``end`` and the vector to ``towards``."""
    coords = list(line.coords)
    if (coords[0][0], coords[0][1]) == (end.x, end.y):
        coords = coords[::-1]            # make ``end`` the last vertex
    # heading over the last ~10 m, not the last vertex pair, which is noisy
    x1, y1 = coords[-1]
    back = coords[-2]
    for c in reversed(coords[:-1]):
        back = c
        if math.hypot(x1 - c[0], y1 - c[1]) >= 10:
            break
    hx, hy = x1 - back[0], y1 - back[1]
    vx, vy = towards.x - x1, towards.y - y1
    n = math.hypot(hx, hy) * math.hypot(vx, vy)
    return (hx * vx + hy * vy) / n if n else -1.0

def suggest_roads(
    confidence_path: Path | str,
    network: dict,
    dismissed: list[list],
    *,
    threshold: float = 0.25,
    min_length_m: float = 30.0,
    covered_share: float = 0.5,
    gap_max_m: float = 30.0,
    min_connected_ends: int = 0,
    limit: int = 60,
) -> dict:
    """Candidate roads the network is probably missing, ranked.

    Two kinds, both label-free:

    * ``missed``: the model was *somewhat* sure there is road here -- above a
      permissive threshold, below the one used for the network -- and nothing
      has been drawn there yet. These are the roads a person is most likely to
      want, and the cheapest to confirm.
    * ``gap``: two dead ends that nearly meet. A real street rarely just stops
      ten metres short of another, so a short connector is usually right.

    Anything the person already dismissed is not offered again.
    """
    import numpy as np
    import rasterio
    import shapely
    from pyproj import Transformer
    from rasterio.features import rasterize
    from scipy import ndimage
    from shapely.geometry import LineString, mapping
    from shapely.ops import transform

    from gisagent.vector.roads import (_douglas_peucker, _sample_confidence,
                                       clean_mask, skeletonize_mask,
                                       trace_skeleton)

    with rasterio.open(confidence_path) as ds:
        conf = ds.read(1).astype(np.float32)
        tf, crs = ds.transform, ds.crs
    fwd = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform
    inv = Transformer.from_crs(crs, "EPSG:4326", always_xy=True).transform
    inv_tf = ~tf
    px = abs(tf.a)

    net_lines = [transform(fwd, LineString(f["geometry"]["coordinates"]))
                 for f in network.get("features", [])
                 if f.get("geometry", {}).get("type") == "LineString"]
    gone_lines = [transform(fwd, LineString(c)) for c in dismissed]

    # "Is this candidate already drawn?" asked of a whole network per candidate
    # is a buffer of hundreds of km of line each time -- minutes. One distance
    # transform answers it for every pixel at once; candidates then just sample.
    def distance_m(lines):
        if not lines:
            return None
        burnt = rasterize([(ln, 1) for ln in lines], out_shape=conf.shape,
                          transform=tf, all_touched=True, dtype="uint8")
        return ndimage.distance_transform_edt(burnt == 0) * px

    net_dist, gone_dist = distance_m(net_lines), distance_m(gone_lines)

    def share_near(line, dist, buffer_m, start=0.0, end=None) -> float:
        if dist is None or line.length == 0:
            return 0.0
        end = line.length if end is None else end
        if end <= start:
            return 0.0
        pts = shapely.line_interpolate_point(
            line, np.linspace(start, end, max(2, int((end - start) / px) + 1)))
        xy = shapely.get_coordinates(pts)
        cols, rows = inv_tf * (xy[:, 0], xy[:, 1])
        rows = np.clip(rows.astype(int), 0, dist.shape[0] - 1)
        cols = np.clip(cols.astype(int), 0, dist.shape[1] - 1)
        return float((dist[rows, cols] <= buffer_m).mean())

    def fresh(line) -> bool:
        return (share_near(line, net_dist, ALONG_BUFFER_M * 1.5) < covered_share
                and share_near(line, gone_dist, ALONG_BUFFER_M) < 0.5)

    out = []

    # missed roads: permissive threshold, minus what is already drawn
    cleaned, _ = clean_mask(conf > threshold, min_object_px=200,
                            min_hole_px=200, close_radius=2)
    for coords in trace_skeleton(skeletonize_mask(cleaned), tf):
        coords = _douglas_peucker(coords, 2.0)
        if len(coords) < 2:
            continue
        line = LineString(coords)
        if line.length < min_length_m or not fresh(line):
            continue
        # A missing street joins the streets around it; a parking lot or a
        # flat roof that merely looks like road usually floats free.
        joined = sum(1 for p in (coords[0], coords[-1])
                     if _dist_at(net_dist, inv_tf, p) <= CONNECT_M)
        if joined < min_connected_ends:
            continue
        c = _sample_confidence(conf, inv_tf, coords)
        out.append(("missed", line, c, c * math.log1p(line.length) * (1 + joined)))

    # gaps: pairs of dead ends that nearly meet
    for a, b, align in _dangle_pairs(net_lines, gap_max_m):
        line = LineString([a, b])
        # A connector touches the network at both ends by construction, so
        # judge only its middle: stay two buffers clear of each end, or the
        # stub it grows out of reads as "already drawn".
        margin = 2 * ALONG_BUFFER_M
        if share_near(line, net_dist, ALONG_BUFFER_M, start=margin,
                      end=line.length - margin) >= covered_share:
            continue
        if share_near(line, gone_dist, ALONG_BUFFER_M) >= 0.5:
            continue
        c = _sample_confidence(conf, inv_tf, list(line.coords))
        out.append(("gap", line, c, align + c))

    # Gaps first: measured on Boston, 80% of the top-ranked gap connectors are
    # real roads against ~45% of "missed" candidates, so a reviewer's first
    # clicks should go where they are most likely to be accepts.
    out.sort(key=lambda r: (r[0] == "gap", r[3]), reverse=True)
    features = []
    for n, (kind, line, c, score) in enumerate(out[:limit]):
        geom = mapping(transform(inv, line))
        features.append({
            "type": "Feature",
            "properties": {"id": f"s-{n}", "kind": kind,
                           "length_m": round(line.length, 1),
                           "confidence": round(float(c), 4),
                           "confidence_pct": round(100 * float(c), 1),
                           "score": round(float(score), 4)},
            "geometry": {"type": "LineString",
                         "coordinates": [list(map(float, p)) for p in geom["coordinates"]]},
        })
    counts = {"missed": sum(1 for f in features if f["properties"]["kind"] == "missed"),
              "gap": sum(1 for f in features if f["properties"]["kind"] == "gap")}
    return {"type": "FeatureCollection", "features": features,
            "stats": {"n": len(features), **counts, "threshold": threshold,
                      "total_length_m": round(sum(f["properties"]["length_m"]
                                                  for f in features), 1)}}
