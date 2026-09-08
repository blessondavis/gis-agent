"""Grid decoding for the Massachusetts Roads tiles.

Region assembly rests on the claim that a tile's filename gives its position in
EPSG:26986. These are the published header values for four tiles spread across
the dataset, so a regression in the offsets or the decode shows up here rather
than as a silently misplaced mosaic.
"""

from __future__ import annotations

import pytest

from gisagent.dataset.mass_roads import (
    GRID_STEP,
    TILE_SIZE_M,
    TileRef,
    decode_name,
    find_contiguous_block,
    largest_connected_group,
)

# name -> Origin reported by gdalinfo for that tile
PUBLISHED_ORIGINS = {
    "10378780_15": (102986.436174005270004, 878771.024778023362160),
    "10828720_15": (107486.436173997819424, 872771.024778019636869),
    "11128870_15": (110486.436174012720585, 887771.024778027087450),
    "17878780_15": (177986.436173997819424, 878771.024778023362160),
}


def ref(name: str, split: str = "test") -> TileRef:
    key_e, key_n = decode_name(name)
    return TileRef(name=name, split=split, key_e=key_e, key_n=key_n)


def test_decode_name_splits_into_easting_and_northing_keys():
    assert decode_name("10378780_15") == (1037, 8780)
    assert decode_name("22529485_15") == (2252, 9485)


@pytest.mark.parametrize("name,expected", PUBLISHED_ORIGINS.items())
def test_origin_matches_published_geotiff_header(name, expected):
    got = ref(name).origin
    assert got == pytest.approx(expected, abs=1e-6)


def test_bounds_span_exactly_one_tile():
    minx, miny, maxx, maxy = ref("10378780_15").bounds
    assert maxx - minx == pytest.approx(TILE_SIZE_M)
    assert maxy - miny == pytest.approx(TILE_SIZE_M)


def test_eastward_neighbour_abuts_with_no_gap_or_overlap():
    """The whole contiguity argument depends on this being exact."""
    a = ref("10378780_15")
    b = TileRef("east", "test", a.key_e + GRID_STEP, a.key_n)
    assert b.bounds[0] == pytest.approx(a.bounds[2])   # b.minx == a.maxx
    assert b.bounds[1] == pytest.approx(a.bounds[1])   # same row


def test_southward_neighbour_abuts():
    a = ref("10378780_15")
    b = TileRef("south", "test", a.key_e, a.key_n - GRID_STEP)
    assert b.bounds[3] == pytest.approx(a.bounds[1])   # b.maxy == a.miny


@pytest.mark.parametrize("bad", ["1234567_15", "abcdefgh_15", "123456789_15"])
def test_decode_name_rejects_malformed_names(bad):
    with pytest.raises(ValueError):
        decode_name(bad)


def _grid(cols: int, rows: int, e0: int = 1000, n0: int = 9000) -> list[TileRef]:
    return [
        TileRef(f"{e0 + c * GRID_STEP:04d}{n0 - r * GRID_STEP}_15", "train",
                e0 + c * GRID_STEP, n0 - r * GRID_STEP)
        for r in range(rows)
        for c in range(cols)
    ]


def test_find_contiguous_block_returns_mutually_adjacent_tiles():
    block = find_contiguous_block(_grid(3, 3), 2, 2)
    assert block is not None and len(block) == 4

    keys = {(t.key_e, t.key_n) for t in block}
    e0 = min(k[0] for k in keys)
    n0 = max(k[1] for k in keys)
    assert keys == {
        (e0, n0), (e0 + GRID_STEP, n0),
        (e0, n0 - GRID_STEP), (e0 + GRID_STEP, n0 - GRID_STEP),
    }


def test_find_contiguous_block_returns_raster_order():
    """Mosaicking assumes top-left first, reading left to right."""
    block = find_contiguous_block(_grid(2, 2), 2, 2)
    assert [(t.key_e, t.key_n) for t in block] == [
        (1000, 9000), (1015, 9000), (1000, 8985), (1015, 8985)
    ]


def test_find_contiguous_block_returns_none_when_region_too_small():
    assert find_contiguous_block(_grid(2, 2), 3, 3) is None


def test_find_contiguous_block_honours_exclude():
    tiles = _grid(2, 2)
    assert find_contiguous_block(tiles, 2, 2) is not None
    assert find_contiguous_block(tiles, 2, 2, exclude={tiles[0].name}) is None


def test_isolated_tiles_do_not_form_a_group():
    far = TileRef("far", "train", 5000, 5000)
    group = largest_connected_group([*_grid(2, 2), far])
    assert len(group) == 4
    assert far not in group


def test_largest_connected_group_prefers_the_bigger_island():
    small = [TileRef("s1", "train", 5000, 5000),
             TileRef("s2", "train", 5000 + GRID_STEP, 5000)]
    group = largest_connected_group([*_grid(3, 3), *small])
    assert len(group) == 9
