"""Massachusetts Roads dataset (Mnih) access.

The tiles are genuine GeoTIFFs in EPSG:26986 (NAD83 / Massachusetts Mainland)
at 1.0 m/px, 1500x1500 px => 1.5 x 1.5 km per tile.

The filename encodes the tile's position on a 100 m grid, which lets us pick a
spatially *contiguous* block of tiles without downloading anything first::

    10378780_15  ->  key_e=1037, key_n=8780
    origin_x = key_e * 100 + X_OFFSET
    origin_y = key_n * 100 + Y_OFFSET

Verified against the published headers of four separate tiles. Because a tile
spans exactly 1500 m and neighbours differ by 15 grid units, adjacent tiles abut
with no gap and no overlap, so a mosaic of them is seamless by construction.
"""

from __future__ import annotations

import concurrent.futures as cf
import re
from dataclasses import dataclass, field
from pathlib import Path

import httpx

BASE_URL = "https://www.cs.toronto.edu/~vmnih/data/mass_roads"
SPLITS = ("train", "valid", "test")

CRS = "EPSG:26986"
TILE_PX = 1500
PIXEL_SIZE_M = 1.0
TILE_SIZE_M = TILE_PX * PIXEL_SIZE_M       # 1500 m
GRID_UNIT_M = 100                          # filename units
GRID_STEP = int(TILE_SIZE_M // GRID_UNIT_M)  # 15 -> adjacency step

# Empirically derived from the published tile headers (constant across tiles).
X_OFFSET = -713.563825994729996
Y_OFFSET = 771.024778023362160

_LINK_RE = re.compile(r"([0-9]{8})_15\.tiff?")


@dataclass(frozen=True)
class TileRef:
    """One dataset tile, with its position on the grid and in the CRS."""

    name: str          # e.g. "10378780_15"
    split: str
    key_e: int         # grid column (units of 100 m)
    key_n: int         # grid row

    @property
    def sat_url(self) -> str:
        return f"{BASE_URL}/{self.split}/sat/{self.name}.tiff"

    @property
    def map_url(self) -> str:
        # Asymmetry in the upstream dataset: satellite is .tiff, label is .tif
        return f"{BASE_URL}/{self.split}/map/{self.name}.tif"

    @property
    def origin(self) -> tuple[float, float]:
        """Top-left corner in EPSG:26986 metres."""
        return (
            self.key_e * GRID_UNIT_M + X_OFFSET,
            self.key_n * GRID_UNIT_M + Y_OFFSET,
        )

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """(minx, miny, maxx, maxy) in EPSG:26986."""
        ox, oy = self.origin
        return (ox, oy - TILE_SIZE_M, ox + TILE_SIZE_M, oy)

    def neighbours(self) -> dict[str, tuple[int, int]]:
        return {
            "east": (self.key_e + GRID_STEP, self.key_n),
            "west": (self.key_e - GRID_STEP, self.key_n),
            "north": (self.key_e, self.key_n + GRID_STEP),
            "south": (self.key_e, self.key_n - GRID_STEP),
        }


def decode_name(name: str) -> tuple[int, int]:
    """Return the grid key for a tile name: '10378780_15' -> (1037, 8780)."""
    stem = name.split("_")[0]
    if len(stem) != 8 or not stem.isdigit():
        raise ValueError(f"unexpected tile name: {name!r}")
    return int(stem[:4]), int(stem[4:])


def list_split(split: str, *, timeout: float = 60.0) -> list[TileRef]:
    """Parse the upstream index page for a split."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    url = f"{BASE_URL}/{split}/sat/index.html"
    resp = httpx.get(url, timeout=timeout, follow_redirects=True)
    resp.raise_for_status()

    refs: list[TileRef] = []
    seen: set[str] = set()
    for digits in _LINK_RE.findall(resp.text):
        name = f"{digits}_15"
        if name in seen:
            continue
        seen.add(name)
        e, n = decode_name(name)
        refs.append(TileRef(name=name, split=split, key_e=e, key_n=n))
    if not refs:
        raise RuntimeError(f"no tiles parsed from {url}")
    return refs


def find_contiguous_block(
    refs: list[TileRef], rows: int = 2, cols: int = 2,
    exclude: set[str] | None = None,
) -> list[TileRef] | None:
    """Find a rows x cols block of tiles that are all spatially adjacent.

    Returned in raster order (top-left first, reading left to right), which is
    what the mosaicking step expects. Names in ``exclude`` are treated as
    missing, which is how unusable tiles are kept out of a block.
    """
    if exclude:
        refs = [r for r in refs if r.name not in exclude]
    by_key = {(r.key_e, r.key_n): r for r in refs}
    for ref in sorted(refs, key=lambda r: (-r.key_n, r.key_e)):
        block: list[TileRef] = []
        ok = True
        for dr in range(rows):
            for dc in range(cols):
                key = (ref.key_e + dc * GRID_STEP, ref.key_n - dr * GRID_STEP)
                found = by_key.get(key)
                if found is None:
                    ok = False
                    break
                block.append(found)
            if not ok:
                break
        if ok and len(block) == rows * cols:
            return block
    return None


def largest_connected_group(refs: list[TileRef]) -> list[TileRef]:
    """Biggest set of tiles connected edge to edge (flood fill over the grid)."""
    by_key = {(r.key_e, r.key_n): r for r in refs}
    unvisited = set(by_key)
    best: list[TileRef] = []
    while unvisited:
        stack = [unvisited.pop()]
        group: list[TileRef] = []
        while stack:
            key = stack.pop()
            ref = by_key[key]
            group.append(ref)
            for nkey in ref.neighbours().values():
                if nkey in unvisited:
                    unvisited.discard(nkey)
                    stack.append(nkey)
        if len(group) > len(best):
            best = group
    return sorted(best, key=lambda r: (-r.key_n, r.key_e))


@dataclass
class TileQuality:
    """How usable a tile is, measured without downloading it in full."""

    name: str
    blank_fraction: float = 0.0   # white no-data padding
    road_fraction: float = 0.0    # ground-truth road coverage
    error: str = ""

    @property
    def usable(self) -> bool:
        return not self.error

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "blank_fraction": round(self.blank_fraction, 4),
            "road_fraction": round(self.road_fraction, 5),
            "error": self.error,
        }


def screen_tiles(
    refs: list[TileRef],
    *,
    max_workers: int = 12,
    sample_px: int = 188,
    cache_path: Path | None = None,
    check_blank: bool = False,
    progress=None,
) -> dict[str, TileQuality]:
    """Rank tiles by road density, over the network, cheaply.

    Only the label rasters are read, and they are fetched with a plain HTTP GET
    rather than through GDAL's /vsicurl. That is not a micro-optimisation:
    measured over 24 tiles, /vsicurl costs 14.5 s per tile against 0.64 s for a
    direct GET -- 23x -- because it issues a HEAD plus ranged reads to probe a
    file that is only 6-10 KB in the first place (a road mask is almost all
    zeros, so it compresses away to nothing).

    ``check_blank`` reads the satellite tile too, and is off by default because
    it is ~1000x more expensive: the satellite tiles are 6.6 MB, carry no
    internal overviews, and are striped one row per block, so a downsampled
    /vsicurl read still has to pull essentially the whole file. Screening a few
    hundred tiles that way means gigabytes of traffic. Blankness is instead
    measured by :func:`measure_blank` after a block is downloaded, where the
    pixels are already local and the check is free.

    Results are cached to ``cache_path``, so an interrupted scan resumes rather
    than refetching.
    """
    import io
    import json
    import warnings

    import rasterio
    from rasterio.enums import Resampling

    cached: dict[str, TileQuality] = {}
    if cache_path and cache_path.exists():
        try:
            raw = json.loads(cache_path.read_text())
            for name, rec in raw.items():
                cached[name] = TileQuality(
                    name=name,
                    blank_fraction=rec.get("blank_fraction", 0.0),
                    road_fraction=rec.get("road_fraction", 0.0),
                    error=rec.get("error", ""),
                )
        except Exception:
            cached = {}  # a corrupt cache is not worth failing over

    # Re-try previously failed tiles; a failure is usually a transient fetch.
    todo = [r for r in refs if r.name not in cached or cached[r.name].error]

    def one(ref: TileRef) -> TileQuality:
        q = TileQuality(name=ref.name)
        shape = (sample_px, sample_px)
        if check_blank:
            try:
                with rasterio.open(f"/vsicurl/{ref.sat_url}") as ds:
                    bands = min(3, ds.count)
                    arr = ds.read(
                        list(range(1, bands + 1)),
                        out_shape=(bands, *shape),
                        resampling=Resampling.average,
                    )
                q.blank_fraction = float((arr.min(axis=0) >= 248).mean())
            except Exception as exc:
                q.error = f"image: {exc}"
                return q
        try:
            resp = client.get(ref.map_url)
            resp.raise_for_status()
            with warnings.catch_warnings():
                # the label rasters carry no CRS of their own; georeferencing
                # comes from the matching satellite tile
                warnings.simplefilter("ignore")
                with rasterio.open(io.BytesIO(resp.content)) as ds:
                    lbl = ds.read(1, out_shape=(1, *shape),
                                  resampling=Resampling.average)
            q.road_fraction = float((lbl > 127).mean())
        except Exception as exc:
            q.error = f"label: {exc}"
        return q

    if todo:
        limits = httpx.Limits(max_connections=max_workers,
                              max_keepalive_connections=max_workers)
        with httpx.Client(timeout=60.0, follow_redirects=True,
                          limits=limits) as client:
            with cf.ThreadPoolExecutor(max_workers=max_workers) as pool:
                for q in pool.map(one, todo):
                    cached[q.name] = q
                    if progress:
                        progress(len(cached), len(todo), q)
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {n: q.to_dict() for n, q in cached.items()}
            tmp = cache_path.with_suffix(".json.part")
            tmp.write_text(json.dumps(payload, indent=1))
            tmp.replace(cache_path)

    wanted = {r.name for r in refs}
    return {n: q for n, q in cached.items() if n in wanted}


def measure_blank(path: Path, sample_px: int = 256) -> float:
    """Fraction of a downloaded tile that is white no-data padding.

    Cheap, because the file is already local. Run this after downloading a
    block: tiles at the edge of the survey area are largely blank, and
    mosaicking those wastes inference on nothing.
    """
    import rasterio
    from rasterio.enums import Resampling

    with rasterio.open(path) as ds:
        bands = min(3, ds.count)
        arr = ds.read(
            list(range(1, bands + 1)),
            out_shape=(bands, sample_px, sample_px),
            resampling=Resampling.average,
        )
    return float((arr.min(axis=0) >= 248).mean())


def find_best_block(
    refs: list[TileRef],
    size: int = 2,
    *,
    max_blank: float = 0.15,
    min_road: float = 0.004,
    search_limit: int = 220,
    cache_path: Path | None = None,
) -> tuple[list[TileRef] | None, dict[str, TileQuality]]:
    """Pick a contiguous block that is actually worth looking at.

    Screens candidate tiles for no-data padding and road density, then returns
    the highest-road-density block whose tiles all pass. Falls back to the
    plain contiguous block if screening rules everything out.
    """
    by_key = {(r.key_e, r.key_n): r for r in refs}

    # candidate anchors whose full block exists at all
    anchors: list[list[TileRef]] = []
    for r in sorted(refs, key=lambda r: (-r.key_n, r.key_e)):
        block = []
        for dr in range(size):
            for dc in range(size):
                found = by_key.get(
                    (r.key_e + dc * GRID_STEP, r.key_n - dr * GRID_STEP)
                )
                if found is None:
                    block = []
                    break
                block.append(found)
            if not block:
                break
        if block:
            anchors.append(block)
        if len(anchors) >= search_limit:
            break

    if not anchors:
        return None, {}

    # screen only the tiles that actually appear in a candidate block
    unique = {t.name: t for blk in anchors for t in blk}
    quality = screen_tiles(list(unique.values()), cache_path=cache_path)

    scored = []
    for blk in anchors:
        qs = [quality.get(t.name) for t in blk]
        if any(q is None or not q.usable for q in qs):
            continue
        # blank_fraction is 0.0 unless screening was asked to fetch imagery;
        # the real blank check happens after download, in measure_blank()
        if max(q.blank_fraction for q in qs) > max_blank:
            continue
        mean_road = sum(q.road_fraction for q in qs) / len(qs)
        if mean_road < min_road:
            continue
        scored.append((mean_road, blk))

    if not scored:
        return find_contiguous_block(refs, size, size), quality
    scored.sort(key=lambda x: -x[0])
    return scored[0][1], quality


def rank_blocks(
    refs: list[TileRef],
    size: int = 2,
    *,
    min_road: float = 0.004,
    search_limit: int = 400,
    cache_path: Path | None = None,
) -> tuple[list[tuple[float, list[TileRef]]], dict[str, TileQuality]]:
    """All candidate blocks, densest first, so a caller can fall through.

    Returned as (mean_road_fraction, tiles). Blank tiles cannot be detected
    without downloading, so callers should verify the top choice with
    :func:`measure_blank` and move to the next entry if it is mostly no-data.
    """
    by_key = {(r.key_e, r.key_n): r for r in refs}
    anchors: list[list[TileRef]] = []
    for r in sorted(refs, key=lambda r: (-r.key_n, r.key_e)):
        block: list[TileRef] = []
        ok = True
        for dr in range(size):
            for dc in range(size):
                found = by_key.get(
                    (r.key_e + dc * GRID_STEP, r.key_n - dr * GRID_STEP)
                )
                if found is None:
                    ok = False
                    break
                block.append(found)
            if not ok:
                break
        if ok and block:
            anchors.append(block)
        if len(anchors) >= search_limit:
            break

    if not anchors:
        return [], {}

    unique = {t.name: t for blk in anchors for t in blk}
    quality = screen_tiles(list(unique.values()), cache_path=cache_path)

    scored = []
    for blk in anchors:
        qs = [quality.get(t.name) for t in blk]
        if any(q is None or not q.usable for q in qs):
            continue
        mean_road = sum(q.road_fraction for q in qs) / len(qs)
        if mean_road < min_road:
            continue
        scored.append((mean_road, blk))
    scored.sort(key=lambda x: -x[0])
    return scored, quality


@dataclass
class DownloadResult:
    ref: TileRef
    image_path: Path
    label_path: Path | None = None
    bytes_downloaded: int = 0
    skipped: bool = False
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def _fetch(client: httpx.Client, url: str, dest: Path) -> int:
    """Download url to dest unless already present. Returns bytes written."""
    if dest.exists() and dest.stat().st_size > 0:
        return 0
    tmp = dest.with_suffix(dest.suffix + ".part")
    written = 0
    with client.stream("GET", url) as resp:
        resp.raise_for_status()
        with tmp.open("wb") as fh:
            for chunk in resp.iter_bytes(chunk_size=1 << 16):
                fh.write(chunk)
                written += len(chunk)
    tmp.replace(dest)
    return written


def download_tiles(
    refs: list[TileRef],
    dest_dir: Path,
    *,
    with_labels: bool = True,
    max_workers: int = 4,
    timeout: float = 300.0,
) -> list[DownloadResult]:
    """Download satellite tiles (and ground-truth road masks) into dest_dir."""
    img_dir = dest_dir / "sat"
    lbl_dir = dest_dir / "map"
    img_dir.mkdir(parents=True, exist_ok=True)
    if with_labels:
        lbl_dir.mkdir(parents=True, exist_ok=True)

    def one(ref: TileRef) -> DownloadResult:
        img = img_dir / f"{ref.name}.tif"
        lbl = (lbl_dir / f"{ref.name}.tif") if with_labels else None
        res = DownloadResult(ref=ref, image_path=img, label_path=lbl)
        pre_existing = img.exists()
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            try:
                res.bytes_downloaded += _fetch(client, ref.sat_url, img)
            except Exception as exc:  # keep going; report per tile
                res.errors.append(f"image: {exc}")
            if lbl is not None:
                try:
                    res.bytes_downloaded += _fetch(client, ref.map_url, lbl)
                except Exception as exc:
                    res.errors.append(f"label: {exc}")
        res.skipped = pre_existing and res.bytes_downloaded == 0
        return res

    with cf.ThreadPoolExecutor(max_workers=max_workers) as pool:
        return list(pool.map(one, refs))
