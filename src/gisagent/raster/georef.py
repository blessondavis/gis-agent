"""Georeferencing repair for the dataset's label rasters.

The Mnih label masks ship with no CRS and an identity transform, while the
matching satellite tiles are properly georeferenced GeoTIFFs. Same pixel grid,
same dimensions, so the fix is to copy the spatial reference across. Without
this, every label is stranded in pixel space: evaluation cannot align it to a
prediction and QGIS cannot place it on a map.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import rasterio


def needs_georeferencing(path: Path | str) -> bool:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with rasterio.open(path) as ds:
            return ds.crs is None or ds.transform.is_identity


def copy_georeferencing(
    source: Path | str,
    target: Path | str,
    out_path: Path | str | None = None,
) -> Path:
    """Stamp target with source's CRS and transform.

    Rewrites in place when out_path is omitted. Raises if the pixel grids differ,
    because silently georeferencing a mismatched raster would put every road in
    the wrong place.
    """
    source, target = Path(source), Path(target)
    out_path = Path(out_path) if out_path else target

    with rasterio.open(source) as src:
        crs, transform = src.crs, src.transform
        shape = (src.height, src.width)

    if crs is None:
        raise ValueError(f"source {source} has no CRS to copy")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with rasterio.open(target) as tgt:
            if (tgt.height, tgt.width) != shape:
                raise ValueError(
                    f"grid mismatch: {target.name} is {tgt.width}x{tgt.height}, "
                    f"{source.name} is {shape[1]}x{shape[0]}"
                )
            data = tgt.read()
            profile = tgt.profile.copy()

    profile.update(
        driver="GTiff",
        crs=crs,
        transform=transform,
        compress="deflate",
        tiled=True,
        blockxsize=256,
        blockysize=256,
    )

    tmp = out_path.with_suffix(".georef.tmp.tif")
    with rasterio.open(tmp, "w", **profile) as dst:
        dst.write(data)
    tmp.replace(out_path)
    return out_path


def georeference_labels(sat_dir: Path | str, map_dir: Path | str) -> dict[str, str]:
    """Repair every label that has a matching satellite tile."""
    sat_dir, map_dir = Path(sat_dir), Path(map_dir)
    report: dict[str, str] = {}
    for label in sorted(map_dir.glob("*.tif")):
        sat = sat_dir / label.name
        if not sat.exists():
            report[label.name] = "skipped: no matching satellite tile"
            continue
        if not needs_georeferencing(label):
            report[label.name] = "already georeferenced"
            continue
        try:
            copy_georeferencing(sat, label)
            report[label.name] = "georeferenced"
        except Exception as exc:
            report[label.name] = f"failed: {exc}"
    return report
