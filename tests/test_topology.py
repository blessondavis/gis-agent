"""Label-free topology scoring.

This is the signal that has to work when there is no ground truth, and it is
also the one that covers the vision critic's blind spot (a uniformly shifted
annotation still looks right to a VLM but stops meeting itself). So the
properties it claims are pinned here.
"""

from __future__ import annotations

import pytest

from gisagent.vector.topology import analyse

M = 1.0 / 111320.0          # roughly one metre in degrees near the equator


def fc(*lines) -> dict:
    return {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "properties": {},
             "geometry": {"type": "LineString", "coordinates": [list(p) for p in ln]}}
            for ln in lines
        ],
    }


def projected(*lines) -> dict:
    """Same, but declaring a projected CRS so coordinates are read as metres."""
    out = fc(*lines)
    out["crs"] = {"type": "name",
                  "properties": {"name": "urn:ogc:def:crs:EPSG::26986"}}
    return out


def test_empty_input_is_reported_not_crashed():
    r = analyse(fc())
    assert r.n_features == 0
    assert r.verdict == "empty"


def test_two_lines_meeting_at_a_point_are_one_component():
    r = analyse(projected(((0, 0), (100, 0)), ((100, 0), (200, 0))))
    assert r.n_components == 1
    assert r.largest_component_share == pytest.approx(1.0)


def test_endpoints_just_inside_tolerance_still_join():
    """The bug this pins: grid quantisation split points 1 m apart.

    Two endpoints 1 m apart must join at a 2.5 m tolerance regardless of where
    they fall relative to any internal grid, so the offset here is deliberately
    chosen to straddle a cell boundary.
    """
    r = analyse(projected(((0, 0), (2.0, 0)), ((3.0, 0), (10.0, 0))), snap_m=2.5)
    assert r.n_components == 1, "1 m gap must be snapped at a 2.5 m tolerance"


def test_endpoints_beyond_tolerance_stay_separate():
    r = analyse(projected(((0, 0), (10, 0)), ((40, 0), (80, 0))), snap_m=2.5)
    assert r.n_components == 2


def test_disconnected_network_scores_below_connected_one():
    joined = projected(((0, 0), (100, 0)), ((100, 0), (200, 0)),
                       ((100, 0), (100, 100)))
    broken = projected(((0, 0), (100, 0)), ((300, 0), (400, 0)),
                       ((700, 500), (800, 600)))
    assert analyse(joined).score > analyse(broken).score
    assert analyse(joined).n_components == 1
    assert analyse(broken).n_components == 3


def test_largest_component_share_reflects_length_not_count():
    r = analyse(projected(
        ((0, 0), (1000, 0)), ((1000, 0), (2000, 0)),   # 2 km joined
        ((9000, 9000), (9010, 9000)),                  # 10 m orphan
    ))
    assert r.n_components == 2
    assert r.largest_component_share > 0.99


def test_junctions_and_dangles_are_counted():
    # a T: three lines meeting at one node, three loose ends
    r = analyse(projected(((0, 0), (100, 0)), ((100, 0), (200, 0)),
                          ((100, 0), (100, 100))))
    assert r.n_junctions == 1
    assert r.n_dangles == 3


def test_short_isolated_fragments_are_flagged():
    r = analyse(projected(
        ((0, 0), (1000, 0)),
        ((5000, 5000), (5005, 5000)),
        ((6000, 6000), (6004, 6000)),
    ), short_fragment_m=40.0)
    assert r.short_fragments == 2


def test_geographic_coordinates_are_handled_in_metres():
    """Longitude degrees are shorter than latitude degrees away from the
    equator; the tolerance must mean the same thing in both axes."""
    lat = 42.35
    r = analyse(fc(
        ((-71.0, lat), (-71.0 + 100 * M, lat)),
        ((-71.0 + 100 * M, lat), (-71.0 + 200 * M, lat)),
    ), snap_m=2.5)
    assert r.n_components == 1
    assert r.total_length_m > 0


def test_score_is_bounded():
    for gj in (projected(((0, 0), (10, 0))),
               projected(*[((i * 500, 0), (i * 500 + 5, 0)) for i in range(40)])):
        assert 0.0 <= analyse(gj).score <= 1.0


def test_multilinestring_is_split_into_parts():
    gj = {"type": "FeatureCollection", "features": [{
        "type": "Feature", "properties": {},
        "geometry": {"type": "MultiLineString",
                     "coordinates": [[[0, 0], [100, 0]], [[500, 500], [600, 500]]]},
    }]}
    r = analyse(gj)
    assert r.n_features == 2
    assert r.n_components == 2
