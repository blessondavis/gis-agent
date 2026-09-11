"""Training-data plumbing used by fine-tuning: discovery, mixing, splitting."""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("torch")

from gisagent.train.data import MassRoadsCrops, TilePair  # noqa: E402
from gisagent.train.loop import split_pairs  # noqa: E402


def _tile(path, value, size=96):
    img = np.full((size, size, 3), value, np.uint8)
    lbl = np.zeros((size, size), np.uint8)
    lbl[40:56, :] = 255
    Image.fromarray(img).save(path["sat"])
    Image.fromarray(lbl).save(path["map"])


@pytest.fixture
def two_domains(tmp_path):
    """Four dark "domain A" tiles and one bright "domain B" tile, as PNGs."""
    for d in ("a", "b"):
        (tmp_path / d / "sat").mkdir(parents=True)
        (tmp_path / d / "map").mkdir(parents=True)
    for i in range(4):
        _tile({"sat": tmp_path / "a/sat" / f"a{i}.png", "map": tmp_path / "a/map" / f"a{i}.png"}, 40)
    _tile({"sat": tmp_path / "b/sat/b0.png", "map": tmp_path / "b/map/b0.png"}, 200)
    return tmp_path


def test_discover_reads_pngs_and_filters_by_id(two_domains):
    a = TilePair.discover(two_domains / "a/sat", two_domains / "a/map")
    assert [p.image.stem for p in a] == ["a0", "a1", "a2", "a3"]
    some = TilePair.discover(two_domains / "a/sat", two_domains / "a/map", ids={"a1", "a3"})
    assert [p.image.stem for p in some] == ["a1", "a3"]


def test_weights_set_the_mix_regardless_of_dataset_size(two_domains):
    a = TilePair.discover(two_domains / "a/sat", two_domains / "a/map")
    b = TilePair.discover(two_domains / "b/sat", two_domains / "b/map")
    # half the crops from the single B tile, half spread over the four A tiles
    ds = MassRoadsCrops(a + b, crop=32, length=400, augment=False, road_bias=0.0,
                        weights=[0.5 / 4] * 4 + [0.5], seed=1)
    bright = Counter(bool(ds[i][0].mean() > 0.5) for i in range(400))
    assert 0.42 < bright[True] / 400 < 0.58

    unweighted = MassRoadsCrops(a + b, crop=32, length=400, augment=False, road_bias=0.0, seed=1)
    assert sum(bool(unweighted[i][0].mean() > 0.5) for i in range(400)) / 400 < 0.3


def test_weights_must_match_pairs(two_domains):
    a = TilePair.discover(two_domains / "a/sat", two_domains / "a/map")
    with pytest.raises(ValueError):
        MassRoadsCrops(a, crop=32, weights=[1.0])


def test_split_is_deterministic_and_disjoint():
    pairs = list(range(135))
    tr, va = split_pairs(pairs, 0.15, seed=0)
    assert (len(tr), len(va)) == (115, 20)
    assert not set(tr) & set(va)
    assert split_pairs(pairs, 0.15, seed=0) == (tr, va)
