"""Cut a georeferenced raster into overlapping chips, and put masks back together.

Two rules drive this module:

1. Every chip carries its own affine transform, so a chip is a valid standalone
   GeoTIFF. Nothing downstream has to remember where a chip came from.
2. Chips overlap, and the masks are recombined with a cosine taper. A model run
   independently per chip disagrees with itself near the seams; averaging across
   the overlap makes those seams disappear instead of leaving a grid of scars
   across the mosaic.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window, transform as window_transform


@dataclass(frozen=True)
class ChipSpec:
    """One chip cut from a source raster."""

    chip_id: str
    source: str
    col_off: int
    row_off: int
    width: int
    height: int
    bounds: tuple[float, float, float, float]   # in source CRS
    crs: str
    path: str | None = None

    @property
    def window(self) -> Window:
        return Window(self.col_off, self.row_off, self.width, self.height)

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "ChipSpec":
        d = dict(d)
        d["bounds"] = tuple(d["bounds"])
        return ChipSpec(**d)


def plan_chips(
    width: int,
    height: int,
    chip_size: int,
    overlap: int,
) -> list[tuple[int, int, int, int]]:
    """Plan chip windows as (col_off, row_off, w, h).

    The last chip in each direction is pulled back flush with the edge rather
    than being left partial, so every chip is exactly chip_size where the source
    is large enough. That keeps the model's input size constant, which matters
    because segmentation quality varies with input scale.
    """
    if chip_size <= 0:
        raise ValueError("chip_size must be positive")
    if not 0 <= overlap < chip_size:
        raise ValueError("overlap must be >= 0 and < chip_size")

    step = chip_size - overlap

    def offsets(total: int) -> list[int]:
        if total <= chip_size:
            return [0]
        n = math.ceil((total - chip_size) / step) + 1
        offs = [min(i * step, total - chip_size) for i in range(n)]
        # de-duplicate while preserving order (the clamp can repeat the last one)
        seen: list[int] = []
        for o in offs:
            if not seen or o != seen[-1]:
                seen.append(o)
        return seen

    out = []
    for row in offsets(height):
        for col in offsets(width):
            out.append(
                (col, row, min(chip_size, width - col), min(chip_size, height - row))
            )
    return out


def tile_raster(
    src_path: Path | str,
    out_dir: Path | str,
    *,
    chip_size: int = 1024,
    overlap: int = 128,
    write_chips: bool = True,
    prefix: str = "chip",
) -> list[ChipSpec]:
    """Cut src_path into overlapping chips; optionally write them as GeoTIFFs."""
    src_path = Path(src_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    specs: list[ChipSpec] = []
    with rasterio.open(src_path) as ds:
        plan = plan_chips(ds.width, ds.height, chip_size, overlap)
        profile_base = ds.profile.copy()

        for col, row, w, h in plan:
            win = Window(col, row, w, h)
            tf = window_transform(win, ds.transform)
            left, top = tf @ (0, 0)
            right, bottom = tf @ (w, h)
            chip_id = f"{prefix}_{row:05d}_{col:05d}"
            path = None

            if write_chips:
                data = ds.read(window=win)
                profile = profile_base.copy()
                profile.update(
                    driver="GTiff",
                    height=h,
                    width=w,
                    transform=tf,
                    compress="deflate",
                    tiled=True,
                    blockxsize=256,
                    blockysize=256,
                )
                dest = out_dir / f"{chip_id}.tif"
                with rasterio.open(dest, "w", **profile) as dst:
                    dst.write(data)
                path = str(dest)

            specs.append(
                ChipSpec(
                    chip_id=chip_id,
                    source=str(src_path),
                    col_off=col,
                    row_off=row,
                    width=w,
                    height=h,
                    bounds=(min(left, right), min(top, bottom),
                            max(left, right), max(top, bottom)),
                    crs=str(ds.crs),
                    path=path,
                )
            )

    (out_dir / "chips.json").write_text(
        json.dumps([s.to_dict() for s in specs], indent=2), encoding="utf-8"
    )
    return specs


def load_chips(out_dir: Path | str) -> list[ChipSpec]:
    data = json.loads((Path(out_dir) / "chips.json").read_text(encoding="utf-8"))
    return [ChipSpec.from_dict(d) for d in data]


def _taper(h: int, w: int, overlap: int) -> np.ndarray:
    """Cosine ramp that falls to ~0 at the chip edge, 1.0 in the interior."""
    if overlap <= 0:
        return np.ones((h, w), dtype=np.float32)

    def ramp(n: int) -> np.ndarray:
        v = np.ones(n, dtype=np.float32)
        k = min(overlap, n // 2)
        if k <= 0:
            return v
        # 0..1 over k samples, never exactly 0 so edge chips still contribute
        t = 0.5 * (1 - np.cos(np.linspace(0, math.pi, k + 2)[1:-1]))
        v[:k] = t
        v[n - k:] = t[::-1]
        return v

    return np.outer(ramp(h), ramp(w)).astype(np.float32)


def stitch_masks(
    chip_masks: dict[str, np.ndarray],
    specs: list[ChipSpec],
    reference: Path | str,
    out_path: Path | str,
    *,
    overlap: int = 128,
    threshold: float | None = 0.5,
    dtype: str = "uint8",
) -> Path:
    """Blend per-chip masks back into one georeferenced raster.

    chip_masks maps chip_id -> float array in [0, 1] (or bool). Overlaps are
    averaged with a cosine taper so seams do not survive into the mosaic.
    """
    reference = Path(reference)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(reference) as ref:
        H, W = ref.height, ref.width
        profile = ref.profile.copy()

    acc = np.zeros((H, W), dtype=np.float32)
    wsum = np.zeros((H, W), dtype=np.float32)

    by_id = {s.chip_id: s for s in specs}
    for chip_id, mask in chip_masks.items():
        spec = by_id.get(chip_id)
        if spec is None:
            raise KeyError(f"no ChipSpec for chip_id {chip_id!r}")
        m = np.asarray(mask, dtype=np.float32)
        if m.shape != (spec.height, spec.width):
            raise ValueError(
                f"{chip_id}: mask shape {m.shape} != chip {(spec.height, spec.width)}"
            )
        wt = _taper(spec.height, spec.width, overlap)
        r, c = spec.row_off, spec.col_off
        acc[r:r + spec.height, c:c + spec.width] += m * wt
        wsum[r:r + spec.height, c:c + spec.width] += wt

    blended = np.divide(acc, wsum, out=np.zeros_like(acc), where=wsum > 0)

    if threshold is not None:
        out = (blended >= threshold).astype(dtype)
        if dtype == "uint8":
            out *= 255
        nodata = 0
    else:
        out = blended.astype("float32")
        nodata = None

    profile.update(
        driver="GTiff",
        count=1,
        dtype=out.dtype.name,
        compress="deflate",
        tiled=True,
        blockxsize=256,
        blockysize=256,
        nodata=nodata,
        photometric="MINISBLACK",
    )
    profile.pop("colorinterp", None)

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(out, 1)
    return out_path


def build_vrt_mosaic(tif_paths: list[Path | str], out_vrt: Path | str) -> Path:
    """Mosaic tiles into a VRT (no pixel copy) using GDAL's Python bindings.

    Falls back to the gdalbuildvrt CLI when the bindings are unavailable, which
    is the common case for a plain `pip install rasterio` environment.
    """
    out_vrt = Path(out_vrt)
    out_vrt.parent.mkdir(parents=True, exist_ok=True)
    paths = [str(p) for p in tif_paths]

    try:
        from osgeo import gdal  # type: ignore

        gdal.BuildVRT(str(out_vrt), paths)
        return out_vrt
    except ImportError:
        pass

    import shutil
    import subprocess

    exe = shutil.which("gdalbuildvrt") or shutil.which("gdalbuildvrt.exe")
    if exe is None:
        raise RuntimeError(
            "neither the GDAL python bindings nor gdalbuildvrt are available"
        )
    subprocess.run([exe, "-q", "-overwrite", str(out_vrt), *paths], check=True)
    return out_vrt
