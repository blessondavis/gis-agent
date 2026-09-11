"""Human edits layered over the machine network, and length-based scoring."""

from __future__ import annotations

import json

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from gisagent import pipeline
from gisagent.checkpoints import Checkpoints
from gisagent.evaluate.network import expected_f1, score_network
from gisagent.vector.edits import EditLog, merge_network

# A small grid near Boston, in degrees. ~0.0001 deg lat is ~11 m.
LAT, LNG = 42.35, -71.06
D = 0.0010                     # ~110 m


def line(*pts):
    return [[LNG + x * D, LAT + y * D] for x, y in pts]


def fc(*lines):
    return {"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {"confidence": 0.9, "confidence_pct": 90},
         "geometry": {"type": "LineString", "coordinates": c}} for c in lines]}


MACHINE = fc(line((0, 0), (4, 0)),        # a long east-west street
             line((2, -2), (2, 0)),       # a side street meeting it
             line((0, 3), (4, 3)))        # a parallel street, 330 m north


@pytest.fixture
def log(tmp_path):
    return EditLog(tmp_path / "edits.json")


def test_no_edits_is_the_machine_network(log):
    net, s = merge_network(MACHINE, log.state())
    assert s.n_model == 3 and s.n_human == 0 and s.human_share == 0


def test_deleting_a_model_road_survives_revectorisation(log):
    log.delete({"source": "model", "id": "m-2", "geometry": line((0, 3), (4, 3))})
    # the agent re-traces the same road a couple of metres off
    retraced = fc(line((0, 0), (4, 0)), line((2, -2), (2, 0)),
                  line((0, 3.02), (4, 3.02)))
    net, s = merge_network(retraced, log.state())
    assert s.n_deleted == 1 and s.n_model == 2


def test_deleting_a_street_keeps_the_street_that_crosses_it(log):
    log.delete({"source": "model", "id": "m-0", "geometry": line((0, 0), (4, 0))})
    net, s = merge_network(MACHINE, log.state())
    assert s.n_deleted == 1 and s.n_model == 2     # the side street survives


def test_a_drawn_road_ending_on_a_street_splits_it_into_a_junction(log):
    # a new street from the north, ending (slightly short) on the middle of m-0
    log.add_line(line((3, 1.5), (3, 0.01)))
    net, s = merge_network(MACHINE, log.state())
    assert s.n_snapped_ends == 1
    assert s.n_model == 4                          # m-0 split in two
    human = next(f for f in net["features"] if f["properties"]["source"] == "human")
    end = human["geometry"]["coordinates"][-1]
    ends = [f["geometry"]["coordinates"][i] for f in net["features"]
            if f["properties"]["source"] == "model" for i in (0, -1)]
    assert sum(1 for e in ends if np.allclose(e, end, atol=1e-9)) == 2


def test_redrawing_a_model_road_supersedes_it(log):
    log.add_line(line((0, 3.01), (4, 3.01)))
    net, s = merge_network(MACHINE, log.state())
    assert s.n_superseded == 1 and s.n_human == 1


def test_undo_redo_and_new_edits_fork_history(log):
    log.add_line(line((5, 5), (6, 6)))
    log.add_line(line((7, 7), (8, 8)))
    assert len(log.state().human) == 2
    log.undo()
    assert len(log.state().human) == 1 and log.state().can_redo
    log.redo()
    assert len(log.state().human) == 2
    log.undo()
    log.add_line(line((9, 9), (9, 10)))            # forks: redo is gone
    assert not log.state().can_redo
    assert len(EditLog(log.path).state().human) == 2     # persisted


def test_replace_and_delete_of_human_roads_use_their_ids(log):
    op = log.add_line(line((5, 5), (6, 6)))
    log.replace({"source": "human", "id": op["id"]}, line((5, 5), (6, 7)))
    st = log.state()
    assert len(st.human) == 1 and op["id"] not in st.human
    log.delete({"source": "human", "id": next(iter(st.human))})
    assert log.state().human == {}


@pytest.mark.parametrize("bad", [[], [[1, 1]], [[0, 0], [0, 0]], [[500, 0], [1, 1]], "x"])
def test_bad_geometry_is_rejected(log, bad):
    with pytest.raises(ValueError):
        log.add_line(bad)


# --------------------------------------------------------------------------- #
# length-based scoring
# --------------------------------------------------------------------------- #

@pytest.fixture
def truth_raster(tmp_path):
    """400 x 400 px at 1 m in EPSG:3857 with one 7 px wide horizontal road."""
    from pyproj import Transformer

    x0, y0 = Transformer.from_crs(4326, 3857, always_xy=True).transform(LNG, LAT)
    tf = from_origin(x0, y0 + 400, 1.0, 1.0)
    a = np.zeros((400, 400), np.uint8)
    a[197:204, :] = 255
    path = tmp_path / "truth.tif"
    with rasterio.open(path, "w", driver="GTiff", width=400, height=400, count=1,
                       dtype="uint8", crs="EPSG:3857", transform=tf) as dst:
        dst.write(a, 1)
    inv = Transformer.from_crs(3857, 4326, always_xy=True).transform

    def road(xs, row):
        return [list(inv(x0 + x, y0 + 400 - row)) for x in xs]
    return path, road, tf


def _net(*lines, source="model"):
    return {"features": [{"properties": {"source": source},
                          "geometry": {"type": "LineString", "coordinates": c}}
                         for c in lines]}


def test_completeness_counts_how_much_road_was_found(truth_raster):
    path, road, _ = truth_raster
    half = score_network(_net(road([0, 200], 200.5)), path)["overall"]
    full = score_network(_net(road([0, 399], 200.5)), path)["overall"]
    assert 0.45 < half["completeness"] < 0.55
    assert full["completeness"] > 0.97 and full["correctness"] > 0.97


def test_edit_gain_is_what_the_human_added(truth_raster):
    path, road, _ = truth_raster
    net = _net(road([0, 200], 200.5))
    net["features"] += _net(road([200, 399], 200.5), source="human")["features"]
    s = score_network(net, path)
    assert s["edit_gain"]["completeness"] > 0.4
    assert s["overall"]["completeness"] > s["model_only"]["completeness"]


def test_a_road_in_the_wrong_place_is_not_correct(truth_raster):
    path, road, _ = truth_raster
    s = score_network(_net(road([0, 399], 60)), path)["overall"]
    assert s["correctness"] < 0.05 and s["completeness"] < 0.05


def test_expected_f1_prefers_the_network_that_matches_confidence(truth_raster):
    path, road, tf = truth_raster
    with rasterio.open(path) as ds:
        conf = (ds.read(1) > 0).astype(np.float32)
        crs = ds.crs
    right = expected_f1(_net(road([0, 399], 200.5)), conf, tf, crs)
    wrong = expected_f1(_net(road([0, 399], 60)), conf, tf, crs)
    assert right > 0.9 > 0.1 > wrong


def test_expected_precision_and_recall_point_the_right_way(truth_raster):
    from gisagent.evaluate.network import expected_scores

    path, road, tf = truth_raster
    with rasterio.open(path) as ds:
        conf = (ds.read(1) > 0).astype(np.float32)
        crs = ds.crs
    half = expected_scores(_net(road([0, 200], 200.5)), conf, tf, crs)
    extra = expected_scores(_net(road([0, 399], 200.5), road([0, 399], 60)), conf, tf, crs)
    # half the road: precise but incomplete; the road plus a false one: complete but imprecise
    assert half["precision"] > 0.9 and half["recall"] < 0.6
    assert extra["recall"] > 0.9 and extra["precision"] < 0.6


def test_f_beta_weights_the_side_the_objective_cares_about():
    from gisagent.evaluate.network import BETA, f_beta

    precise, complete = (0.9, 0.5), (0.5, 0.9)          # (precision, recall)
    assert f_beta(*precise, BETA["precision"]) > f_beta(*complete, BETA["precision"])
    assert f_beta(*complete, BETA["recall"]) > f_beta(*precise, BETA["recall"])
    assert f_beta(*precise, 1.0) == pytest.approx(f_beta(*complete, 1.0))


def test_candidate_variants_are_validated_not_trusted():
    from gisagent.pipeline import Job

    # the two inputs a live model actually sent
    assert Job.normalise_variant({"mask_threshold": 0.45}) == {"threshold": 0.45}
    assert Job.normalise_variant({"close_radius": 0.5})["close_radius"] == 0
    assert Job.normalise_variant({})["threshold"] == 0.5
    for bad in ({"thresh": 0.4}, {"threshold": 1.2}, {"close_radius": "wide"}, [0.4]):
        with pytest.raises(ValueError):
            Job.normalise_variant(bad)


# --------------------------------------------------------------------------- #
# checkpoints
# --------------------------------------------------------------------------- #

def test_checkpoint_restores_files_including_copy_on_write_ones(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "jobs_root", lambda: tmp_path)
    job = pipeline.new_job("cp")
    (job.dir / "roads.geojson").write_text("v1")
    job.conf_dir.mkdir()
    np.save(job.conf_dir / "chip_a.npy", np.zeros(4))

    cps = Checkpoints(job.dir)
    cid = cps.begin("turn one", message_index=0)
    # the turn: overwrite a chip (preserved first), create one, change roads
    job._preserve(job.conf_dir / "chip_a.npy")
    np.save(job.conf_dir / "chip_a.npy", np.ones(4))
    job._preserve(job.conf_dir / "chip_b.npy")
    np.save(job.conf_dir / "chip_b.npy", np.ones(4))
    (job.dir / "roads.geojson").write_text("v2")
    (job.dir / "edits.json").write_text("{}")
    cps.end()

    cps.restore(cid)
    assert (job.dir / "roads.geojson").read_text() == "v1"
    assert np.load(job.conf_dir / "chip_a.npy").sum() == 0
    assert not (job.conf_dir / "chip_b.npy").exists()
    assert not (job.dir / "edits.json").exists()
    assert json.loads((job.dir / "job.json").read_text())["name"] == "cp"


def test_restoring_an_older_checkpoint_unwinds_later_turns(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "jobs_root", lambda: tmp_path)
    job = pipeline.new_job("cp2")
    job.conf_dir.mkdir()
    np.save(job.conf_dir / "c.npy", np.full(2, 1.0))
    cps = Checkpoints(job.dir)

    first = cps.begin("one", 0)
    cps.end()                                       # turn one changed nothing
    cps.begin("two", 2)
    job._preserve(job.conf_dir / "c.npy")
    np.save(job.conf_dir / "c.npy", np.full(2, 2.0))
    cps.end()

    cps.restore(first)
    assert np.load(job.conf_dir / "c.npy")[0] == 1.0
    assert [c["id"] for c in cps.list()] == [first]
