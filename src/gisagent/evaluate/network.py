"""Score a centreline network against ground truth, by length.

Pixel IoU answers "how well does the mask overlap the labels". Once a person is
finishing the network by hand, the question changes to "how much of the road
network is there": a road drawn two pixels off-centre is still a finished road.
The standard answer from the road-extraction literature (Wiedemann et al.) is a
pair of length-based measures with a tolerance buffer:

* completeness -- share of the true road length within the buffer of the
  network. "What fraction of the roads have we got?"
* correctness  -- share of the network length within the buffer of a true
  road. "What fraction of what we drew is real?"
* quality      -- matched truth / (truth + unmatched network), one number.

Both are computed on rasters at the truth resolution: the network is burned in
as 1 px lines, the truth mask is thinned to its centreline, and the buffer is a
Euclidean distance transform, so the tolerance is exact rather than a blocky
dilation.
"""

from __future__ import annotations

import numpy as np


def _burn(features: list[dict], shape, transform, crs) -> np.ndarray:
    from pyproj import Transformer
    from rasterio.features import rasterize
    from shapely.geometry import LineString
    from shapely.ops import transform as shp_transform

    fwd = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform
    shapes = []
    for f in features:
        g = f.get("geometry") or {}
        if g.get("type") == "LineString" and len(g.get("coordinates", [])) >= 2:
            shapes.append((shp_transform(fwd, LineString(g["coordinates"])), 1))
    if not shapes:
        return np.zeros(shape, bool)
    return rasterize(shapes, out_shape=shape, transform=transform,
                     all_touched=True, dtype="uint8").astype(bool)


def _scores(pred: np.ndarray, truth_skel: np.ndarray, tol_px: float,
            truth_near: np.ndarray) -> dict:
    from scipy import ndimage

    truth_len = int(truth_skel.sum())
    pred_len = int(pred.sum())
    if pred_len:
        near_pred = ndimage.distance_transform_edt(~pred) <= tol_px
        matched_truth = int((truth_skel & near_pred).sum())
    else:
        matched_truth = 0
    matched_pred = int((pred & truth_near).sum())
    unmatched_pred = pred_len - matched_pred
    return {
        "completeness": round(matched_truth / truth_len, 4) if truth_len else 0.0,
        "correctness": round(matched_pred / pred_len, 4) if pred_len else 0.0,
        "quality": (round(matched_truth / (truth_len + unmatched_pred), 4)
                    if truth_len + unmatched_pred else 0.0),
        "network_px": pred_len,
    }


BETA = {"balanced": 1.0, "precision": 0.5, "recall": 2.0}


def f_beta(precision: float, recall: float, beta: float) -> float:
    b2 = beta * beta
    denom = b2 * precision + recall
    return (1 + b2) * precision * recall / denom if denom else 0.0


def expected_scores(network: dict, conf: np.ndarray, transform, crs, *,
                    half_width_m: float = 3.5) -> dict:
    """Expected precision / recall / F1 of a network under the confidence map.

    precision = confidence inside the network's footprint / footprint area;
    recall    = confidence inside the footprint / all confidence.
    See :func:`expected_f1` for when this is (and is not) a fair judge.
    """
    from scipy import ndimage

    lines = _burn(network.get("features", []), conf.shape, transform, crs)
    if not lines.any():
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    px = abs(transform.a)
    band = ndimage.distance_transform_edt(~lines) <= half_width_m / px
    inside = float(conf[band].sum())
    p = inside / float(band.sum())
    r = inside / float(conf.sum()) if conf.sum() else 0.0
    return {"precision": p, "recall": r, "f1": f_beta(p, r, 1.0)}


def expected_f1(network: dict, conf: np.ndarray, transform, crs, *,
                half_width_m: float = 3.5) -> float:
    """F1 the network would score if the confidence map were calibrated truth.

    A label-free judge for choosing *between post-processing variants of one
    model's output*: burn the centrelines back in at label width, then count
    confidence inside (expected true positives) against confidence everywhere
    (expected road). On the Boston region its ranking of seven threshold /
    vectoriser variants matched ground-truth quality at Spearman 0.96, where
    topology scoring ranked them backwards (-0.64).

    It is self-referential: it trusts the model's confidence, so it cannot
    see a road the model is confidently wrong about. Use it to pick settings,
    not to judge the model.
    """
    return expected_scores(network, conf, transform, crs,
                           half_width_m=half_width_m)["f1"]


def score_network(network: dict, truth_path, *, tolerance_m: float = 5.0) -> dict:
    """Completeness / correctness / quality of a network, overall and by source.

    ``by_source.model`` is what the model delivered on its own;
    ``overall`` includes the person's edits. The difference is what the edits
    were worth.
    """
    import rasterio
    from skimage.morphology import skeletonize

    with rasterio.open(truth_path) as ds:
        truth = ds.read(1) > 127
        transform, crs = ds.transform, ds.crs
        px = abs(transform.a)

    from scipy import ndimage

    truth_skel = skeletonize(truth)
    tol_px = tolerance_m / px
    truth_near = ndimage.distance_transform_edt(~truth_skel) <= tol_px

    feats = network.get("features", [])
    model = [f for f in feats if (f.get("properties") or {}).get("source", "model") == "model"]
    overall = _scores(_burn(feats, truth.shape, transform, crs),
                      truth_skel, tol_px, truth_near)
    by_model = _scores(_burn(model, truth.shape, transform, crs),
                       truth_skel, tol_px, truth_near)
    return {
        "tolerance_m": tolerance_m,
        "truth_length_m": round(float(truth_skel.sum()) * px, 1),
        "overall": overall,
        "model_only": by_model,
        "edit_gain": {
            k: round(overall[k] - by_model[k], 4)
            for k in ("completeness", "correctness", "quality")
        },
    }
