"""Browser-ready previews of georeferenced rasters.

Leaflet places an image overlay by lat/lon corners and stretches it linearly, so
handing it a raster still in EPSG:26986 would put the roads visibly off the
imagery. Everything is therefore warped to EPSG:4326 first and the true warped
bounds are returned alongside the PNG.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling

WGS84 = "EPSG:4326"


def _warp_to_wgs84(path: Path, bands: list[int], max_dim: int = 2048):
    """Reproject selected bands to WGS84, capped at max_dim on the long edge."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with rasterio.open(path) as ds:
            transform, width, height = calculate_default_transform(
                ds.crs, WGS84, ds.width, ds.height, *ds.bounds
            )
            scale = min(1.0, max_dim / max(width, height))
            if scale < 1.0:
                width = max(1, int(width * scale))
                height = max(1, int(height * scale))
                transform, width, height = calculate_default_transform(
                    ds.crs, WGS84, ds.width, ds.height, *ds.bounds,
                    dst_width=width, dst_height=height,
                )

            out = np.zeros((len(bands), height, width), dtype=ds.dtypes[0])
            for i, b in enumerate(bands):
                reproject(
                    source=rasterio.band(ds, b),
                    destination=out[i],
                    src_transform=ds.transform,
                    src_crs=ds.crs,
                    dst_transform=transform,
                    dst_crs=WGS84,
                    resampling=Resampling.bilinear,
                )

    west, north = transform * (0, 0)
    east, south = transform * (width, height)
    # Leaflet wants [[south, west], [north, east]]
    bounds = [[float(south), float(west)], [float(north), float(east)]]
    return out, bounds


def render_image_preview(src: Path | str, dest: Path | str,
                         max_dim: int = 2048) -> dict:
    """RGB satellite imagery -> PNG in WGS84."""
    from PIL import Image

    src, dest = Path(src), Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(src) as ds:
        bands = [1, 2, 3] if ds.count >= 3 else [1]
    arr, bounds = _warp_to_wgs84(src, bands, max_dim)

    if arr.shape[0] == 1:
        arr = np.repeat(arr, 3, axis=0)
    rgb = np.transpose(arr, (1, 2, 0)).astype(np.uint8)
    Image.fromarray(rgb).save(dest, optimize=True)
    return {"url_path": dest.name, "bounds": bounds,
            "width": rgb.shape[1], "height": rgb.shape[0]}


def render_mask_overlay(src: Path | str, dest: Path | str,
                        colour: tuple[int, int, int] = (255, 45, 85),
                        alpha: int = 190, max_dim: int = 2048) -> dict:
    """Binary mask -> transparent RGBA PNG so it can sit over the imagery."""
    from PIL import Image

    src, dest = Path(src), Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    arr, bounds = _warp_to_wgs84(src, [1], max_dim)
    mask = arr[0]
    on = mask > (127 if mask.dtype == np.uint8 else 0.5)

    h, w = on.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[on, 0], rgba[on, 1], rgba[on, 2] = colour
    rgba[on, 3] = alpha
    Image.fromarray(rgba, mode="RGBA").save(dest, optimize=True)
    return {"url_path": dest.name, "bounds": bounds,
            "coverage": round(float(on.mean()), 5)}


def build_job_previews(job, max_dim: int = 2048) -> dict:
    """Render every preview a job can offer; missing inputs are skipped."""
    out: dict = {}
    job.preview_dir.mkdir(parents=True, exist_ok=True)

    if job.image_path.exists():
        out["image"] = render_image_preview(
            job.image_path, job.preview_dir / "image.png", max_dim
        )
    if job.mask_path.exists():
        out["mask"] = render_mask_overlay(
            job.mask_path, job.preview_dir / "mask.png",
            colour=(255, 45, 85), max_dim=max_dim,
        )
    if job.truth_path.exists():
        out["truth"] = render_mask_overlay(
            job.truth_path, job.preview_dir / "truth.png",
            colour=(0, 200, 120), max_dim=max_dim,
        )
    return out


def render_chip_thumbnail(chip_path: Path | str, dest: Path | str,
                          size: int = 256) -> dict:
    """Small plain-pixel thumbnail of a chip, for the input gallery."""
    from PIL import Image

    chip_path, dest = Path(chip_path), Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with rasterio.open(chip_path) as ds:
            bands = min(3, ds.count)
            arr = ds.read(list(range(1, bands + 1)),
                          out_shape=(bands, size, size),
                          resampling=Resampling.average)
    if arr.shape[0] == 1:
        arr = np.repeat(arr, 3, axis=0)
    Image.fromarray(np.transpose(arr, (1, 2, 0)).astype(np.uint8)).save(dest)
    return {"url_path": dest.name, "size": size}
