"""Assemble a training set from the Massachusetts Roads tiles.

The screening cache already records road density for every tile, so the
training set can be chosen rather than taken at random. Two filters matter:

* Reject tiles below a road-density floor. A tile with almost no road
  contributes almost nothing but still costs 9 MB of disk and a slot in memory.
* Reject the very densest tiles too. Those are downtown blocks where the label
  raster paints wide swathes, and over-representing them skews the model
  towards predicting road everywhere.

Disk is the real constraint: each satellite tile is ~9 MB.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from gisagent.dataset.mass_roads import (
    TileRef,
    TileQuality,
    download_tiles,
    list_split,
    screen_tiles,
)


@dataclass
class FetchReport:
    requested: int
    downloaded: int
    failed: list[str]
    bytes_total: int
    sat_dir: Path
    map_dir: Path

    def to_dict(self) -> dict:
        return {
            "requested": self.requested,
            "downloaded": self.downloaded,
            "failed": self.failed,
            "megabytes": round(self.bytes_total / 1e6, 1),
            "sat_dir": str(self.sat_dir),
            "map_dir": str(self.map_dir),
        }


def select_training_tiles(
    quality: dict[str, TileQuality],
    refs: list[TileRef],
    *,
    n: int = 160,
    min_road: float = 0.02,
    max_road: float = 0.18,
) -> list[TileRef]:
    """Pick n tiles spanning the useful part of the road-density range."""
    by_name = {r.name: r for r in refs}
    usable = [
        (q.road_fraction, by_name[name])
        for name, q in quality.items()
        if q.usable and name in by_name and min_road <= q.road_fraction <= max_road
    ]
    if not usable:
        raise RuntimeError(
            "no tiles passed the density filter; widen min_road/max_road"
        )
    usable.sort(key=lambda x: x[0])
    if len(usable) <= n:
        return [r for _, r in usable]

    # even stride across the sorted range, so the set spans sparse rural
    # through moderately dense suburban rather than clustering at one end
    stride = len(usable) / n
    return [usable[min(int(i * stride), len(usable) - 1)][1] for i in range(n)]


def fetch_training_set(
    dest: Path,
    *,
    split: str = "train",
    n: int = 160,
    min_road: float = 0.02,
    max_road: float = 0.18,
    cache_path: Path | None = None,
    max_workers: int = 6,
) -> FetchReport:
    refs = list_split(split)
    quality = screen_tiles(refs, cache_path=cache_path)
    chosen = select_training_tiles(
        quality, refs, n=n, min_road=min_road, max_road=max_road
    )

    dest = Path(dest)
    results = download_tiles(chosen, dest, with_labels=True, max_workers=max_workers)

    from gisagent.raster.georef import georeference_labels

    georeference_labels(dest / "sat", dest / "map")

    return FetchReport(
        requested=len(chosen),
        downloaded=sum(1 for r in results if r.ok),
        failed=[r.ref.name for r in results if not r.ok],
        bytes_total=sum(r.bytes_downloaded for r in results),
        sat_dir=dest / "sat",
        map_dir=dest / "map",
    )
