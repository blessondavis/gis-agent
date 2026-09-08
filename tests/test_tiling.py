"""Correctness of the chip/stitch round trip.

If stitching is not exact, every accuracy number downstream is measuring the
mosaicker rather than the model, so these are the load-bearing tests.
"""

from __future__ import annotations

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from gisagent.raster.tiling import (
    plan_chips,
    tile_raster,
    stitch_masks,
    load_chips,
)

CRS = "EPSG:26986"


def _write_raster(path, arr, *, origin=(224486.436, 949271.025), res=1.0):
    count = 1 if arr.ndim == 2 else arr.shape[0]
    data = arr[None] if arr.ndim == 2 else arr
    with rasterio.open(
        path, "w", driver="GTiff", height=data.shape[1], width=data.shape[2],
        count=count, dtype=data.dtype.name, crs=CRS,
        transform=from_origin(origin[0], origin[1], res, res),
    ) as dst:
        dst.write(data)
    return path


# --------------------------------------------------------------------------- #
# planning
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "w,h,size,ov",
    [(3000, 3000, 1024, 128), (1500, 1500, 512, 64), (1000, 700, 1024, 128),
     (2048, 1024, 1024, 0), (3000, 1500, 768, 256)],
)
def test_chips_cover_every_pixel(w, h, size, ov):
    plan = plan_chips(w, h, size, ov)
    covered = np.zeros((h, w), dtype=bool)
    for col, row, cw, ch in plan:
        covered[row:row + ch, col:col + cw] = True
        assert col >= 0 and row >= 0
        assert col + cw <= w and row + ch <= h
    assert covered.all(), "chip plan left uncovered pixels"


def test_full_size_chips_when_source_is_large_enough():
    plan = plan_chips(3000, 3000, 1024, 128)
    assert all(cw == 1024 and ch == 1024 for _, _, cw, ch in plan)


def test_overlap_must_be_smaller_than_chip():
    with pytest.raises(ValueError):
        plan_chips(100, 100, 64, 64)


# --------------------------------------------------------------------------- #
# chip geo-referencing
# --------------------------------------------------------------------------- #

def test_chip_bounds_match_source_geography(tmp_path):
    src = _write_raster(tmp_path / "src.tif",
                        np.zeros((3, 1500, 1500), dtype="uint8"))
    specs = tile_raster(src, tmp_path / "chips", chip_size=512, overlap=64)

    with rasterio.open(src) as ds:
        sb = ds.bounds

    # union of chip bounds must equal the source bounds
    assert min(s.bounds[0] for s in specs) == pytest.approx(sb.left)
    assert min(s.bounds[1] for s in specs) == pytest.approx(sb.bottom)
    assert max(s.bounds[2] for s in specs) == pytest.approx(sb.right)
    assert max(s.bounds[3] for s in specs) == pytest.approx(sb.top)

    # each written chip is independently valid and correctly placed
    for spec in specs:
        with rasterio.open(spec.path) as c:
            assert str(c.crs) == CRS
            assert (c.bounds.left, c.bounds.bottom, c.bounds.right, c.bounds.top) \
                == pytest.approx(spec.bounds)


def test_chips_json_round_trips(tmp_path):
    src = _write_raster(tmp_path / "src.tif", np.zeros((3, 800, 800), dtype="uint8"))
    specs = tile_raster(src, tmp_path / "chips", chip_size=256, overlap=32)
    assert load_chips(tmp_path / "chips") == specs


# --------------------------------------------------------------------------- #
# stitching
# --------------------------------------------------------------------------- #

def _roads(h, w):
    """A synthetic road-like mask: thin lines, the hard case for blending."""
    m = np.zeros((h, w), dtype=np.float32)
    m[h // 3: h // 3 + 7, :] = 1.0          # horizontal road
    m[:, w // 2: w // 2 + 5] = 1.0          # vertical road
    for i in range(min(h, w)):              # diagonal
        m[i, i] = 1.0
    return m


def test_stitch_reconstructs_the_original_mask_exactly(tmp_path):
    """Chip a known mask, feed the true sub-masks back, expect the original."""
    h = w = 1600
    truth = _roads(h, w)
    src = _write_raster(tmp_path / "src.tif", (truth * 255).astype("uint8"))

    specs = tile_raster(src, tmp_path / "chips", chip_size=512, overlap=128,
                        write_chips=False)
    chip_masks = {
        s.chip_id: truth[s.row_off:s.row_off + s.height,
                         s.col_off:s.col_off + s.width]
        for s in specs
    }

    out = stitch_masks(chip_masks, specs, src, tmp_path / "out.tif",
                       overlap=128, threshold=0.5)
    with rasterio.open(out) as ds:
        got = ds.read(1) > 127

    assert np.array_equal(got, truth > 0.5), "stitched mask != original mask"


def test_stitch_preserves_georeferencing(tmp_path):
    src = _write_raster(tmp_path / "src.tif", np.zeros((900, 900), dtype="uint8"))
    specs = tile_raster(src, tmp_path / "c", chip_size=512, overlap=128,
                        write_chips=False)
    masks = {s.chip_id: np.zeros((s.height, s.width), np.float32) for s in specs}
    out = stitch_masks(masks, specs, src, tmp_path / "o.tif", overlap=128)

    with rasterio.open(src) as a, rasterio.open(out) as b:
        assert a.transform == b.transform
        assert a.crs == b.crs
        assert (a.height, a.width) == (b.height, b.width)
        assert b.count == 1


def test_disagreement_in_overlap_is_averaged_not_seamed(tmp_path):
    """Two chips that disagree should blend, never leave a hard edge."""
    src = _write_raster(tmp_path / "src.tif", np.zeros((512, 1024), dtype="uint8"))
    specs = tile_raster(src, tmp_path / "c", chip_size=512, overlap=256,
                        write_chips=False)
    assert len(specs) >= 2

    masks = {}
    for i, s in enumerate(specs):
        masks[s.chip_id] = np.full((s.height, s.width), 1.0 if i == 0 else 0.0,
                                   dtype=np.float32)

    out = stitch_masks(masks, specs, src, tmp_path / "o.tif", overlap=256,
                       threshold=None)
    with rasterio.open(out) as ds:
        blended = ds.read(1)

    mid = blended[256, :]
    # a hard seam shows up as an abrupt jump; blending keeps steps small
    assert np.abs(np.diff(mid)).max() < 0.2, "hard seam detected in overlap"


def test_stitch_rejects_wrong_shape(tmp_path):
    src = _write_raster(tmp_path / "src.tif", np.zeros((600, 600), dtype="uint8"))
    specs = tile_raster(src, tmp_path / "c", chip_size=256, overlap=32,
                        write_chips=False)
    bad = {specs[0].chip_id: np.zeros((10, 10), np.float32)}
    with pytest.raises(ValueError):
        stitch_masks(bad, specs, src, tmp_path / "o.tif")
