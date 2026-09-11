"""Global-Scale road dataset (Yin et al., CVPR 2025) as road-segmentation masks.

Why this dataset: our road model was trained only on Massachusetts Roads, one
state of US suburbia. Global-Scale covers six continents at the same 1 m/px
GSD (2048x2048 tiles from Google imagery, OSM road graphs), plus an
out-of-domain test set from cities absent from its training pool (Hong Kong,
Shenzhen, Lucerne). That makes it a cheap way to measure -- and then close --
the geographic generalisation gap without changing resolution.

Why we re-rasterise: Global-Scale ships *graphs*; its ``region_N_gt.png`` is a
~3 px line, far thinner than the Massachusetts labels our model learned from.
We measured the Massachusetts masks (``measure_label_width``): road
cross-sections have a sharp modal width of **7 px** (43% of axis-aligned runs;
2*mean distance-transform at the skeleton = 6.85, area/skeleton length = 7.27).
A 1-px centreline dilated to every pixel within 3 px (Euclidean) reproduces
both statistics (7.03 / 7.28), so ``rasterise_graph(width=7)`` does exactly
that. Using the same label width keeps fine-tuning from teaching the model a
new notion of "how wide is a road" along with the new geography.

Graph format (verified): ``region_N_graph_gt.pickle`` is a dict mapping a node
``(row, col)`` float tuple to a list of neighbour nodes; every edge appears in
both directions and endpoints may lie outside the tile. Their ``gt.png`` is
exactly ``cv2.line(..., (int(col), int(row)), ..., thickness=2)`` -- we
reproduce it at IoU 0.999, and drawing with the axes swapped scores ~0.04.

The official split is *not* the mirror's folder layout. The HF mirror
(``gaetanbahl/Global-Scale-Road-Dataset``) keeps the authors' layout, where
``train/`` holds train, val and in-domain test together; the partition lives
in ``globalscale_data_partition()`` of the authors' ``dataset.py``
(github.com/earth-insights/samroadplus)::

    train/region_{0..2374}     train
    train/region_{2375..2713}  val   (== val/region_{i-2375}, byte-identical)
    train/region_{2714..3337}  in-domain test (== in-domain-test/region_{i-2714})
    out_of_domain/region_{0..129}

Two leakage problems inside that official split are handled here:

* 71 official *train* tiles are byte-identical to a val or test tile (plus 56
  train-train duplicate pairs). Found via the files' LFS sha256.
* Tiles overlap spatially across splits: the val/test tiles were drawn from
  the same sites as train, often shifted by a fraction of a tile. We link
  tiles whose road graphs share junctions under a common translation
  (``link_regions``) and treat each connected component as one *site*. Our
  in-domain test sample is taken from sites that contribute nothing to our
  fine-tuning set, and the fine-tune validation split holds out whole sites.

No per-region city/continent metadata exists (the paper only has a world map),
so train/test regions carry ``city=None``. Region indices are, however,
collected site by site -- consecutive indices are usually adjacent tiles --
so spreading picks evenly over the index range spreads them over sites.
"""

from __future__ import annotations

import collections
import json
import os
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

REPO_ID = "gaetanbahl/Global-Scale-Road-Dataset"
IMAGE_PX = 2048
PIXEL_SIZE_M = 1.0
MASS_LABEL_WIDTH_PX = 7        # measured, see module docstring

# Official partition (samroadplus dataset.py), as indices into mirror train/.
TRAIN_RANGE = range(0, 2375)
VAL_RANGE = range(2375, 2714)
TEST_RANGE = range(2714, 3338)
OOD_RANGE = range(0, 130)

# Out-of-domain cities. The data carries no location metadata; the paper says
# only "Hong Kong, Shenzhen and Lucerne". Assigned by inspecting the imagery,
# so treat these as a well-founded guess (see OOD_CITY_NOTE).
OOD_CITIES: dict[str, tuple[range, ...]] = {
    # Swiss Mittelland farmland, the Reuss, Lake Lucerne: unmistakable
    "Lucerne": (range(1, 39),),
    # Kwai Tsing container port, HK Island hills, Yuen Long fish ponds, the
    # Shenzhen River border (112 straddles it)
    "Hong Kong": (range(0, 1), range(104, 114)),
    # dense urban villages, factory blocks, new districts
    "Shenzhen": (range(39, 104), range(114, 130)),
}
OOD_CITY_NOTE = ("not recorded in the dataset; assigned by inspecting the "
                 "imagery. Lucerne (1-38) is certain; the Hong Kong / Shenzhen "
                 "boundary is a visual judgement and may be off for a few "
                 "tiles (both are Asia either way)")
CITY_CONTINENT = {"Hong Kong": "Asia", "Shenzhen": "Asia", "Lucerne": "Europe"}

Graph = dict[tuple[float, float], list[tuple[float, float]]]


# --------------------------------------------------------------------------
# paths and the official split
# --------------------------------------------------------------------------

def hf_path(folder: str, index: int, kind: str) -> str:
    """Repo path of one file: ``hf_path("train", 123, "sat.png")``.

    The mirror shards each split into folders of 100 tiles (``1/`` = 0..99).
    """
    return f"{folder}/{index // 100 + 1}/region_{index}_{kind}"


def official_split(index: int) -> str:
    """Official split of a mirror ``train/`` index: train / val / test."""
    if index in TRAIN_RANGE:
        return "train"
    if index in VAL_RANGE:
        return "val"
    if index in TEST_RANGE:
        return "test"
    raise ValueError(f"index {index} is outside the train/ folder (0..3337)")


def official_alias(index: int) -> str | None:
    """The same tile's path in the mirror's val/ or in-domain-test/ folder."""
    split = official_split(index)
    if split == "val":
        return hf_path("val", index - VAL_RANGE.start, "sat.png")
    if split == "test":
        return hf_path("in-domain-test", index - TEST_RANGE.start, "sat.png")
    return None


def ood_city(index: int) -> str | None:
    for city, ranges in OOD_CITIES.items():
        if any(index in rng for rng in ranges):
            return city
    return None


# --------------------------------------------------------------------------
# graphs -> masks
# --------------------------------------------------------------------------

def load_graph(path: Path) -> Graph:
    with open(path, "rb") as fh:
        return pickle.load(fh)


def graph_segments(graph: Graph) -> np.ndarray:
    """Unique undirected edges as an (N, 2, 2) array of (row, col) points."""
    seen: set[tuple] = set()
    segs = []
    for a, nbrs in graph.items():
        for b in nbrs:
            key = (a, b) if a <= b else (b, a)
            if key not in seen:
                seen.add(key)
                segs.append(key)
    return np.asarray(segs, dtype=np.float64).reshape(-1, 2, 2)


def centreline(graph: Graph, size: int = IMAGE_PX) -> np.ndarray:
    """1-px road centrelines, drawn at sub-pixel precision. bool (size, size)."""
    import cv2

    shift = 4                                   # 1/16 px fixed point
    canvas = np.zeros((size, size), np.uint8)
    for (r0, c0), (r1, c1) in graph_segments(graph):
        # cv2 wants (x, y) = (col, row); it clips to the canvas itself
        p0 = (int(round(c0 * (1 << shift))), int(round(r0 * (1 << shift))))
        p1 = (int(round(c1 * (1 << shift))), int(round(r1 * (1 << shift))))
        cv2.line(canvas, p0, p1, 1, 1, cv2.LINE_8, shift)
    return canvas.astype(bool)


def rasterise_graph(
    graph: Graph | Path | str,
    width: int = MASS_LABEL_WIDTH_PX,
    size: int = IMAGE_PX,
) -> np.ndarray:
    """Road mask (uint8 0/255) with lines ``width`` px wide.

    Every pixel within (width-1)/2 px of a centreline is road, so the
    cross-section is ``width`` px at any orientation (cv2's thick lines are
    not: its ``thickness=5`` draws 7 px axis-aligned but ~7.3 on diagonals).
    """
    from scipy import ndimage as ndi

    if not isinstance(graph, dict):
        graph = load_graph(Path(graph))
    line = centreline(graph, size)
    if not line.any():
        return np.zeros((size, size), np.uint8)
    dist = ndi.distance_transform_edt(~line)
    return np.where(dist <= (width - 1) / 2, 255, 0).astype(np.uint8)


def reproduce_gt_png(graph: Graph, size: int = IMAGE_PX,
                     swap_axes: bool = False) -> np.ndarray:
    """The dataset's own thin ``gt.png`` recipe (bool), for validation only."""
    import cv2

    canvas = np.zeros((size, size), np.uint8)
    for a, nbrs in graph.items():
        for b in nbrs:
            pa, pb = (a, b) if swap_axes else ((a[1], a[0]), (b[1], b[0]))
            cv2.line(canvas, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])),
                     255, 2)
    return canvas > 0


def iou(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(bool), b.astype(bool)
    union = (a | b).sum()
    return float((a & b).sum() / union) if union else 1.0


def validate_axis_order(graph_path: Path, gt_png: Path) -> dict[str, float]:
    """IoU of our rendering against the shipped gt.png, both axis orders.

    ``ours_w3_*`` is our pipeline thinned to 3 px; ``recipe_*`` is their
    exact recipe. The (row, col) variants must win by a wide margin.
    """
    import cv2

    graph = load_graph(graph_path)
    gt = cv2.imread(str(gt_png), cv2.IMREAD_GRAYSCALE) > 127
    swapped = {(c, r): [(q[1], q[0]) for q in nb]
               for (r, c), nb in graph.items()}
    return {
        "ours_w3_rowcol": iou(rasterise_graph(graph, 3) > 0, gt),
        "ours_w3_xy": iou(rasterise_graph(swapped, 3) > 0, gt),
        "recipe_rowcol": iou(reproduce_gt_png(graph), gt),
        "recipe_xy": iou(reproduce_gt_png(graph, swap_axes=True), gt),
    }


def measure_label_width(mask: np.ndarray) -> dict[str, float]:
    """Estimate the line width of a binary road mask, three ways.

    ``run_mode`` (the most common length of horizontal/vertical road runs) is
    the width of axis-aligned roads; the other two are orientation-averaged
    and read ~0.4 px high/low for a true 7 px line.
    """
    from scipy import ndimage as ndi
    from skimage.morphology import skeletonize

    m = mask.astype(bool)
    skel = skeletonize(m)
    if not skel.any():
        return {"edt": 0.0, "area_per_skeleton": 0.0, "run_mode": 0.0}
    dist = ndi.distance_transform_edt(m)
    runs: collections.Counter[int] = collections.Counter()
    for arr in (m, m.T):
        padded = np.pad(arr.astype(np.int8), ((0, 0), (1, 1)))
        starts = np.argwhere(np.diff(padded, axis=1) == 1)
        ends = np.argwhere(np.diff(padded, axis=1) == -1)
        runs.update((ends[:, 1] - starts[:, 1]).tolist())
    return {
        "edt": float(2 * dist[skel].mean()),
        "area_per_skeleton": float(m.sum() / skel.sum()),
        "run_mode": float(runs.most_common(1)[0][0]),
    }


def blank_fraction(rgb: np.ndarray) -> float:
    """Fraction of pixels that are flat white or black no-data."""
    lo, hi = rgb.min(axis=2), rgb.max(axis=2)
    return float(((lo >= 248) | (hi <= 5)).mean())


# --------------------------------------------------------------------------
# duplicates and spatial sites
# --------------------------------------------------------------------------

def fetch_file_index(cache: Path | None = None) -> dict[str, str]:
    """``{repo path: sha256}`` for every sat/graph file, cached to ``cache``."""
    if cache and cache.exists():
        return json.loads(cache.read_text())
    from huggingface_hub import HfApi

    info = HfApi().dataset_info(REPO_ID, files_metadata=True)
    index = {
        s.rfilename: (s.lfs.sha256 if s.lfs else s.blob_id)
        for s in info.siblings
        if s.rfilename.endswith(("_sat.png", "_graph_gt.pickle"))
    }
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(index))
    return index


def exact_duplicates(file_index: dict[str, str]) -> list[list[int]]:
    """Groups of train/ indices whose satellite PNGs are byte-identical."""
    by_hash: dict[str, list[int]] = collections.defaultdict(list)
    for path, sha in file_index.items():
        parts = path.split("/")
        if parts[0] == "train" and path.endswith("_sat.png"):
            by_hash[sha].append(int(parts[2].split("_")[1]))
    return [sorted(v) for v in by_hash.values() if len(v) > 1]


def graph_fingerprints(graph: Graph) -> dict[tuple, tuple[float, float]]:
    """Translation-invariant keys for junctions (degree >= 3).

    A junction's key is its neighbour offsets rounded to whole pixels. The
    same junction seen from two overlapping or abutting tiles gets the same
    key, and the difference of its positions is the tiles' relative offset.
    Keys that repeat within one tile (regular grids) are dropped as ambiguous.
    """
    out: dict[tuple, list] = {}
    for a, nbrs in graph.items():
        if len(nbrs) >= 3:
            key = tuple(sorted((round(b[0] - a[0]), round(b[1] - a[1]))
                               for b in nbrs))
            out.setdefault(key, []).append(a)
    return {k: v[0] for k, v in out.items() if len(v) == 1}


def link_regions(
    graphs: dict[int, Graph], min_votes: int = 2, max_key_regions: int = 50,
) -> list[tuple[int, int, tuple[int, int], int]]:
    """Pairs of regions that share junctions under one consistent offset.

    Returns (a, b, (drow, dcol), votes). Two junction matches agreeing on the
    offset is already strong evidence: a chance agreement needs two distinct
    junction shapes to coincide at the same translation.
    """
    prints = {i: graph_fingerprints(g) for i, g in graphs.items()}
    inverted: dict[tuple, list[int]] = collections.defaultdict(list)
    for i, fp in prints.items():
        for key in fp:
            inverted[key].append(i)
    votes: dict[tuple[int, int], collections.Counter] = \
        collections.defaultdict(collections.Counter)
    for key, regions in inverted.items():
        if len(regions) > max_key_regions:     # generic shape, uninformative
            continue
        for x, a in enumerate(regions):
            for b in regions[x + 1:]:
                pa, pb = prints[a][key], prints[b][key]
                votes[(a, b)][(round(pb[0] - pa[0]), round(pb[1] - pa[1]))] += 1
    links = []
    for (a, b), counter in votes.items():
        offset, n = counter.most_common(1)[0]
        if n >= min_votes:
            links.append((a, b, offset, n))
    return links


def spatial_groups(indices: Iterable[int],
                   pairs: Iterable[tuple[int, int]]) -> dict[int, int]:
    """Union-find: index -> site id (the smallest index in its component)."""
    parent = {i: i for i in indices}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in pairs:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    return {i: find(i) for i in parent}


# --------------------------------------------------------------------------
# selection
# --------------------------------------------------------------------------

def spread_pick(candidates: list[int], n: int, group_of: dict[int, int],
                cap: int, taken: dict[int, int] | None = None) -> list[int]:
    """Pick ``n`` of ``candidates`` evenly over index order, <= cap per site.

    Deterministic: each target position takes the nearest unused candidate
    whose site is not yet full.
    """
    cands = sorted(candidates)
    taken = collections.Counter(taken or {})
    used: set[int] = set()
    picks: list[int] = []
    if not cands or n <= 0:
        return picks
    for target in np.linspace(0, len(cands) - 1, n):
        t = int(round(target))
        for step in range(len(cands)):
            hit = None
            for pos in (t + step, t - step):
                if 0 <= pos < len(cands) and pos not in used \
                        and taken[group_of[cands[pos]]] < cap:
                    hit = pos
                    break
            if hit is not None:
                used.add(hit)
                picks.append(cands[hit])
                taken[group_of[cands[hit]]] += 1
                break
    return sorted(picks)


@dataclass
class Selection:
    ood: list[int]
    ft_train: list[int]
    ft_val: list[int]
    id_test: list[int]


def select_regions(
    group_of: dict[int, int],
    duplicates: list[list[int]],
    *,
    n_ft: int = 220,
    n_val: int = 30,
    n_id_test: int = 60,
    val_blocks: int = 6,
    val_margin: int = 15,
    cap_per_site: int = 4,
) -> Selection:
    """Choose the region sets. See the module docstring for the leakage rules.

    1. In-domain test: ``n_id_test`` official test tiles, one per site, spread
       over the index range.
    2. Pool: official train tiles, minus byte-duplicates of any val/test tile,
       minus the later copy of train-train duplicates, minus every site that
       holds one of the chosen test tiles.
    3. Validation: ``val_blocks`` runs of consecutive pool picks (index order
       follows collection order, so a run is one area); every site they touch
       and every index within ``val_margin`` of a run is withheld from train.
    4. Train: the remaining ``n_ft - n_val``, spread evenly, <= cap per site.
    """
    exclude: set[int] = set()
    for grp in duplicates:
        if any(official_split(i) != "train" for i in grp):
            exclude.update(i for i in grp if official_split(i) == "train")
        else:
            exclude.update(grp[1:])

    id_test = spread_pick(list(TEST_RANGE), n_id_test, group_of, cap=1)
    test_sites = {group_of[i] for i in id_test}

    pool = [i for i in TRAIN_RANGE
            if i not in exclude and group_of[i] not in test_sites]
    provisional = spread_pick(pool, n_ft, group_of, cap_per_site)

    per_block = max(1, n_val // val_blocks)
    starts = np.linspace(0, len(provisional) - per_block, val_blocks + 2)[1:-1]
    ft_val: list[int] = []
    for s in starts:
        ft_val.extend(provisional[int(round(s)):int(round(s)) + per_block])
    ft_val = sorted(set(ft_val))

    val_sites = {group_of[i] for i in ft_val}
    windows = [(provisional[int(round(s))] - val_margin,
                provisional[int(round(s)) + per_block - 1] + val_margin)
               for s in starts]
    pool_train = [i for i in pool
                  if group_of[i] not in val_sites
                  and not any(lo <= i <= hi for lo, hi in windows)]
    ft_train = spread_pick(pool_train, n_ft - len(ft_val), group_of,
                           cap_per_site)
    return Selection(ood=list(OOD_RANGE), ft_train=ft_train, ft_val=ft_val,
                     id_test=id_test)


# --------------------------------------------------------------------------
# download + conversion
# --------------------------------------------------------------------------

@dataclass
class RegionRecord:
    id: str
    set: str                 # ood / ft_train / ft_val / id_test
    source: str              # repo path of the satellite tile
    official_split: str      # train / val / test / ood
    official_index: int
    alias: str | None        # same tile under val/ or in-domain-test/
    site: int | None         # spatial component (train/ folder numbering)
    city: str | None
    continent: str | None
    road_fraction: float
    blank_fraction: float
    sat: str
    map: str


def region_id(folder: str, index: int) -> str:
    return f"ood_{index:03d}" if folder == "out_of_domain" else f"gs_{index:04d}"


def _download(repo_path: str, raw_dir: Path, cache_dir: Path | None) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(REPO_ID, repo_path, repo_type="dataset",
                                local_dir=raw_dir, cache_dir=cache_dir))


def download_graphs(folder: str, indices: Iterable[int], raw_dir: Path,
                    cache_dir: Path | None = None,
                    max_workers: int = 16) -> dict[int, Path]:
    """Fetch graph pickles (a few hundred KB each), skipping ones present."""
    import concurrent.futures as cf

    indices = list(indices)

    def one(i: int) -> tuple[int, Path]:
        local = raw_dir / hf_path(folder, i, "graph_gt.pickle")
        if local.exists() and local.stat().st_size > 0:
            return i, local
        return i, _download(hf_path(folder, i, "graph_gt.pickle"), raw_dir,
                            cache_dir)

    with cf.ThreadPoolExecutor(max_workers=max_workers) as pool:
        return dict(pool.map(one, indices))


def convert_region(folder: str, index: int, out_dir: Path, raw_dir: Path,
                   *, width: int = MASS_LABEL_WIDTH_PX,
                   cache_dir: Path | None = None) -> tuple[Path, Path]:
    """Download one tile's sat + graph, write sat/<id>.png and map/<id>.png.

    Idempotent: a region whose two outputs exist is left alone. The graph
    pickle is deleted once its mask is written.
    """
    import cv2

    rid = region_id(folder, index)
    sat_out = out_dir / "sat" / f"{rid}.png"
    map_out = out_dir / "map" / f"{rid}.png"
    if sat_out.exists() and map_out.exists():
        return sat_out, map_out
    sat_out.parent.mkdir(parents=True, exist_ok=True)
    map_out.parent.mkdir(parents=True, exist_ok=True)

    graph_local = raw_dir / hf_path(folder, index, "graph_gt.pickle")
    if not graph_local.exists():
        graph_local = _download(hf_path(folder, index, "graph_gt.pickle"),
                                raw_dir, cache_dir)
    if not map_out.exists():
        mask = rasterise_graph(graph_local, width)
        tmp = map_out.with_name(map_out.stem + ".part.png")
        cv2.imwrite(str(tmp), mask)
        os.replace(tmp, map_out)
    if not sat_out.exists():
        sat_local = _download(hf_path(folder, index, "sat.png"), raw_dir,
                              cache_dir)
        os.replace(sat_local, sat_out)          # as-is, same volume: a rename
    graph_local.unlink(missing_ok=True)
    return sat_out, map_out


def _stats(sat: Path, mask: Path) -> tuple[float, float]:
    import cv2

    rgb = cv2.imread(str(sat), cv2.IMREAD_COLOR)
    lbl = cv2.imread(str(mask), cv2.IMREAD_GRAYSCALE)
    return float((lbl > 127).mean()), blank_fraction(rgb)


def build_sites(raw_dir: Path, index_dir: Path,
                duplicates: list[list[int]],
                cache_dir: Path | None = None) -> dict[int, int]:
    """Site id for every train/ index, from road-graph overlaps. Cached.

    Needs all 3338 graph pickles once (~580 MB, deleted afterwards by
    ``prepare``); the resulting links are cached as a small JSON. Byte
    duplicates are merged into one site as well.
    """
    links_path = index_dir / "train_links.json"
    if links_path.exists():
        links = json.loads(links_path.read_text())
    else:
        paths = download_graphs("train", range(TEST_RANGE.stop), raw_dir,
                                cache_dir)
        graphs = {i: load_graph(p) for i, p in paths.items()}
        links = [list(map(int, (a, b, *off, n)))
                 for a, b, off, n in link_regions(graphs)]
        index_dir.mkdir(parents=True, exist_ok=True)
        links_path.write_text(json.dumps(links))
    pairs = [(a, b) for a, b, *_ in links]
    pairs += [(g[0], other) for g in duplicates for other in g[1:]]
    return spatial_groups(range(TEST_RANGE.stop), pairs)


def prepare(
    out_dir: Path,
    *,
    n_ft: int = 220,
    n_val: int = 30,
    n_id_test: int = 60,
    width: int = MASS_LABEL_WIDTH_PX,
    cache_dir: Path | None = None,
    validate: int = 2,
    max_workers: int = 6,
    progress=None,
) -> dict:
    """Select, download and convert everything; write ``manifest.json``.

    Only ``_sat.png`` and ``_graph_gt.pickle`` are fetched (plus ``_gt.png``
    for ``validate`` regions per set, to check the axis order; deleted after).
    Re-running skips regions already converted and reuses cached stats.
    """
    import shutil

    out_dir = Path(out_dir)
    raw_dir = out_dir / "_raw"
    index_dir = out_dir / "_index"
    file_index = fetch_file_index(index_dir / "hf_files.json")
    duplicates = exact_duplicates(file_index)
    group_of = build_sites(raw_dir, index_dir, duplicates, cache_dir)
    sel = select_regions(group_of, duplicates, n_ft=n_ft, n_val=n_val,
                         n_id_test=n_id_test)

    manifest_path = out_dir / "manifest.json"
    old = {}
    if manifest_path.exists():
        old = {r["id"]: r for r in json.loads(manifest_path.read_text())["regions"]}

    jobs = [("out_of_domain", i, "ood") for i in sel.ood]
    jobs += [("train", i, "ft_train") for i in sel.ft_train]
    jobs += [("train", i, "ft_val") for i in sel.ft_val]
    jobs += [("train", i, "id_test") for i in sel.id_test]

    import concurrent.futures as cf

    def convert(job: tuple[str, int, str]) -> tuple[Path, Path]:
        return convert_region(job[0], job[1], out_dir, raw_dir, width=width,
                              cache_dir=cache_dir)

    with cf.ThreadPoolExecutor(max_workers=max_workers) as pool:
        outputs = []
        for n, res in enumerate(pool.map(convert, jobs), start=1):
            outputs.append(res)
            if progress:
                progress(n, len(jobs), res[0].stem)

    records: list[RegionRecord] = []
    for (folder, idx, set_name), (sat, mask) in zip(jobs, outputs):
        rid = region_id(folder, idx)
        if rid in old and old[rid].get("set") == set_name:
            road, blank = old[rid]["road_fraction"], old[rid]["blank_fraction"]
        else:
            road, blank = _stats(sat, mask)
        is_ood = folder == "out_of_domain"
        records.append(RegionRecord(
            id=rid, set=set_name, source=hf_path(folder, idx, "sat.png"),
            official_split="ood" if is_ood else official_split(idx),
            official_index=idx,
            alias=None if is_ood else official_alias(idx),
            site=None if is_ood else group_of[idx],
            city=ood_city(idx) if is_ood else None,
            continent=CITY_CONTINENT.get(ood_city(idx) or "") if is_ood else None,
            road_fraction=round(road, 5), blank_fraction=round(blank, 5),
            sat=str(sat.relative_to(out_dir)).replace("\\", "/"),
            map=str(mask.relative_to(out_dir)).replace("\\", "/"),
        ))

    validation = _validate_sample(jobs, out_dir, raw_dir, cache_dir, validate,
                                  old_manifest=manifest_path)
    counts = collections.Counter(r.set for r in records)
    manifest = {
        "source": f"https://huggingface.co/datasets/{REPO_ID}",
        "official_split": "samroadplus dataset.py globalscale_data_partition(): "
                          "train/ 0-2374 train, 2375-2713 val, 2714-3337 test",
        "label": f"road graph re-rasterised, {width} px wide (0/255)",
        "ood_city_note": OOD_CITY_NOTE,
        "counts": dict(counts),
        "axis_order_validation": validation,
        "regions": [asdict(r) for r in records],
    }
    tmp = manifest_path.with_suffix(".json.part")
    tmp.write_text(json.dumps(manifest, indent=1))
    os.replace(tmp, manifest_path)

    # graphs of unselected regions were only needed for the site analysis
    shutil.rmtree(raw_dir, ignore_errors=True)
    return manifest


def _validate_sample(jobs: list[tuple[str, int, str]], out_dir: Path,
                     raw_dir: Path, cache_dir: Path | None, per_set: int,
                     old_manifest: Path) -> list[dict]:
    """Axis-order check on a few regions per set, against their gt.png."""
    if old_manifest.exists():
        prev = json.loads(old_manifest.read_text()).get("axis_order_validation")
        if prev:
            return prev
    by_set: dict[str, list[tuple[str, int]]] = collections.defaultdict(list)
    for folder, idx, set_name in jobs:
        by_set[set_name].append((folder, idx))
    results = []
    for set_name, items in by_set.items():
        step = max(1, len(items) // max(per_set, 1))
        for folder, idx in items[::step][:per_set]:
            graph = _download(hf_path(folder, idx, "graph_gt.pickle"), raw_dir,
                              cache_dir)
            gt = _download(hf_path(folder, idx, "gt.png"), raw_dir, cache_dir)
            scores = validate_axis_order(graph, gt)
            results.append({"id": region_id(folder, idx), "set": set_name,
                            **{k: round(v, 4) for k, v in scores.items()}})
            gt.unlink(missing_ok=True)
            graph.unlink(missing_ok=True)
    return results
