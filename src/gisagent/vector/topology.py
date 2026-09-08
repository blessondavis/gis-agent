"""Label-free quality signals from the shape of the road network itself.

A real road network is *connected*. Streets meet at junctions, dead ends are
rare, and the whole thing is one graph rather than two hundred loose pieces. So
an annotation can be judged without any ground truth by asking whether it looks
like a road network at all:

* many disconnected components  -> roads were missed between them
* many dangling ends            -> spurs and stubs, or streets cut short
* very short isolated fragments -> noise the vectoriser did not clean up

This matters because it is exactly what the vision critic cannot see. A
uniformly misregistered annotation still *looks* like roads over roads and
scored a middling 50 despite an IoU of 0.127 (docs/vlm-critic.md) -- but shifted
geometry stops meeting itself, so its topology degrades measurably.

Computed on the geometry directly rather than through qgis_process: it is a
graph problem, it needs no CRS transformation, and going out to a subprocess per
call would cost seconds for something that takes milliseconds. The QGIS
algorithms remain available for *repairing* what this measures --
``grass:v.clean`` with rmdangle, ``native:extendlines`` to bridge gaps.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

# Endpoints within this distance are treated as the same junction. At 1 m/px a
# couple of metres is well inside the width of a road, so this joins streets
# that genuinely meet without merging ones that merely pass nearby.
SNAP_M = 2.5


@dataclass
class TopologyReport:
    n_features: int = 0
    total_length_m: float = 0.0
    n_components: int = 0
    largest_component_share: float = 0.0
    n_nodes: int = 0
    n_dangles: int = 0
    dangle_ratio: float = 0.0
    n_junctions: int = 0
    short_fragments: int = 0
    isolated_length_m: float = 0.0
    score: float = 0.0
    verdict: str = ""
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        for k in ("total_length_m", "isolated_length_m"):
            d[k] = round(d[k], 1)
        for k in ("largest_component_share", "dangle_ratio", "score"):
            d[k] = round(d[k], 4)
        return d


class _Union:
    """Union-find over endpoints.

    Endpoints are grouped by *distance*, not by grid cell. Quantising to a grid
    looks equivalent and is not: two endpoints 1 m apart that straddle a cell
    boundary land in different cells and never join, which fragments a network
    that is actually connected. The grid here is only a lookup accelerator --
    each point is compared against its own cell and the eight around it, and
    joined when it is genuinely within tolerance.
    """

    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, a: int) -> int:
        while self.parent[a] != a:
            self.parent[a] = self.parent[self.parent[a]]
            a = self.parent[a]
        return a

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def _cluster_endpoints(points: list[tuple[float, float]], snap: float) -> list[int]:
    """Map each endpoint to a node id, joining any pair within ``snap`` metres."""
    uf = _Union(len(points))
    cells: dict[tuple[int, int], list[int]] = {}
    for i, (x, y) in enumerate(points):
        cells.setdefault((int(x // snap), int(y // snap)), []).append(i)

    snap_sq = snap * snap
    for (cx, cy), members in cells.items():
        neighbours: list[int] = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                neighbours.extend(cells.get((cx + dx, cy + dy), ()))
        for i in members:
            xi, yi = points[i]
            for j in neighbours:
                if j <= i:
                    continue
                xj, yj = points[j]
                if (xi - xj) ** 2 + (yi - yj) ** 2 <= snap_sq:
                    uf.union(i, j)

    roots: dict[int, int] = {}
    out = []
    for i in range(len(points)):
        r = uf.find(i)
        if r not in roots:
            roots[r] = len(roots)
        out.append(roots[r])
    return out


def _lines(geojson: dict):
    for feat in geojson.get("features", []):
        geom = feat.get("geometry") or {}
        t = geom.get("type")
        if t == "LineString":
            yield geom["coordinates"]
        elif t == "MultiLineString":
            yield from geom["coordinates"]


def _length(coords, geographic: bool) -> float:
    """Length in metres. Geographic input is converted locally."""
    total = 0.0
    for (x0, y0, *_), (x1, y1, *_) in zip(coords, coords[1:]):
        if geographic:
            # local equirectangular approximation: fine over a few km
            mlat = math.radians((y0 + y1) / 2.0)
            dx = (x1 - x0) * 111320.0 * math.cos(mlat)
            dy = (y1 - y0) * 110540.0
        else:
            dx, dy = x1 - x0, y1 - y0
        total += math.hypot(dx, dy)
    return total


def _is_geographic(geojson: dict, lines: list) -> bool:
    """Decide whether coordinates are degrees or metres.

    Prefer the declared CRS. Bounds alone are not enough to tell: a projected
    network a few hundred metres across sits comfortably inside +/-180, +/-90
    and would be mistaken for degrees, inflating every length by 111320.
    """
    crs = geojson.get("crs")
    if isinstance(crs, dict):
        name = str(crs.get("properties", {}).get("name", "")).upper()
        if name:
            return ("CRS84" in name or "4326" in name
                    or name.endswith(":WGS 84") or "EPSG::4326" in name)

    # No declared CRS: fall back to bounds, and require the span to look like
    # degrees rather than a small projected extent.
    xs = [p[0] for c in lines for p in c]
    ys = [p[1] for c in lines for p in c]
    if not xs:
        return False
    if max(map(abs, xs)) > 180 or max(map(abs, ys)) > 90:
        return False
    span = max(max(xs) - min(xs), max(ys) - min(ys))
    return span < 10.0 and max(map(abs, xs)) > 0.01


def analyse(
    geojson: dict, *, snap_m: float = SNAP_M, short_fragment_m: float = 40.0,
    geographic: bool | None = None,
) -> TopologyReport:
    """Measure how much the annotation looks like a real road network."""
    rep = TopologyReport()

    lines = [c for c in _lines(geojson) if c and len(c) >= 2]
    rep.n_features = len(lines)
    if not lines:
        rep.verdict = "empty"
        rep.notes.append("no line features")
        return rep

    sample = lines[0][0]
    if geographic is None:
        geographic = _is_geographic(geojson, lines)

    lengths = [_length(c, geographic) for c in lines]
    rep.total_length_m = sum(lengths)

    # Work in metres so the tolerance means the same thing in both axes. A
    # degree of longitude is only cos(lat) as long as a degree of latitude --
    # about 0.74 at this latitude -- so treating them alike would snap loosely
    # east-west and tightly north-south.
    if geographic:
        lat0 = sample[1]
        mx = 111320.0 * math.cos(math.radians(lat0))
        my = 110540.0
    else:
        mx = my = 1.0

    points: list[tuple[float, float]] = []
    for coords in lines:
        points.append((coords[0][0] * mx, coords[0][1] * my))
        points.append((coords[-1][0] * mx, coords[-1][1] * my))

    node_of = _cluster_endpoints(points, snap_m)

    # --- build the node graph -------------------------------------------- #
    adj: dict[int, set[int]] = {}
    ends: list[tuple[int, int]] = []
    for i in range(len(lines)):
        a, b = node_of[2 * i], node_of[2 * i + 1]
        ends.append((a, b))
        adj.setdefault(a, set()).add(i)
        adj.setdefault(b, set()).add(i)

    rep.n_nodes = len(adj)
    degree = {node: len(edges) for node, edges in adj.items()}
    rep.n_dangles = sum(1 for d in degree.values() if d == 1)
    rep.n_junctions = sum(1 for d in degree.values() if d >= 3)
    rep.dangle_ratio = rep.n_dangles / max(rep.n_nodes, 1)

    # --- connected components over the line graph ------------------------- #
    seen: set[int] = set()
    components: list[list[int]] = []
    for start in range(len(lines)):
        if start in seen:
            continue
        stack, group = [start], []
        seen.add(start)
        while stack:
            i = stack.pop()
            group.append(i)
            for node in ends[i]:
                for j in adj.get(node, ()):
                    if j not in seen:
                        seen.add(j)
                        stack.append(j)
        components.append(group)

    rep.n_components = len(components)
    comp_len = [sum(lengths[i] for i in g) for g in components]
    rep.largest_component_share = (
        max(comp_len) / rep.total_length_m if rep.total_length_m else 0.0
    )
    rep.short_fragments = sum(
        1 for g, L in zip(components, comp_len)
        if len(g) == 1 and L < short_fragment_m
    )
    rep.isolated_length_m = sum(
        L for g, L in zip(components, comp_len) if len(g) == 1
    )

    # --- roll up ---------------------------------------------------------- #
    # Connectivity dominates: a fragmented network is the clearest sign that
    # roads were missed. Dangles matter but real networks do have cul-de-sacs,
    # so the penalty is gentler and only bites past a third of all nodes.
    conn = rep.largest_component_share
    dangle_pen = max(0.0, (rep.dangle_ratio - 0.33) / 0.67)
    frag_pen = min(1.0, rep.short_fragments / max(rep.n_features, 1) * 3.0)
    rep.score = max(0.0, min(1.0, 0.65 * conn + 0.35 * (1 - dangle_pen) - 0.15 * frag_pen))

    if rep.score >= 0.75:
        rep.verdict = "coherent"
    elif rep.score >= 0.5:
        rep.verdict = "fragmented"
    elif rep.score >= 0.25:
        rep.verdict = "poor"
    else:
        rep.verdict = "incoherent"

    if rep.n_components > 1:
        rep.notes.append(
            f"{rep.n_components} disconnected pieces; the largest holds "
            f"{conn*100:.0f}% of the length - roads are likely missing between them"
        )
    if rep.dangle_ratio > 0.5:
        rep.notes.append(
            f"{rep.n_dangles} of {rep.n_nodes} nodes are dead ends - "
            "expect spurs or streets cut short"
        )
    if rep.short_fragments:
        rep.notes.append(
            f"{rep.short_fragments} isolated fragments under {short_fragment_m:.0f} m"
        )
    if not rep.notes:
        rep.notes.append("network is connected and free of obvious defects")
    return rep


def analyse_file(path: Path | str, **kw) -> TopologyReport:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return analyse(data, **kw)
