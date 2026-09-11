"""Score a segmenter over a set of labelled tiles, for cross-dataset comparison.

Comparing one model across datasets is mostly a fight with label conventions:
Massachusetts labels are ~7 px wide centreline rasters, other datasets draw
3 px lines or whole road surfaces. Pixel IoU moves a lot with that choice and
very little with whether the roads were actually found. So alongside pixel
IoU/F1 this reports two measures that do not care how wide a label is:

* relaxed precision / recall (the Mnih convention: a few pixels of slack);
* centreline completeness / correctness: both masks are thinned to their
  skeletons and compared by length, within a tolerance.

Everything is micro-averaged -- counts are summed over tiles before dividing
-- so a tile with almost no road cannot swing the result.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable

import numpy as np


def _read_rgb(path) -> np.ndarray:
    import rasterio

    with rasterio.open(path) as ds:
        a = ds.read([1, 2, 3]) if ds.count >= 3 else np.repeat(ds.read(1)[None], 3, 0)
    return np.ascontiguousarray(a.transpose(1, 2, 0)).astype(np.uint8)


def _read_mask(path) -> np.ndarray:
    import rasterio

    with rasterio.open(path) as ds:
        return ds.read(1) > 127


def evaluate_tiles(
    predict: Callable[[np.ndarray], np.ndarray],
    pairs: Iterable,
    *,
    threshold: float = 0.5,
    slack_px: int = 3,
    tolerance_px: float = 5.0,
    progress: Callable[[int, str], None] | None = None,
) -> dict:
    """``predict`` maps an HxWx3 uint8 image to an HxW road probability."""
    import warnings

    from scipy import ndimage
    from skimage.morphology import skeletonize

    from gisagent.vector.roads import clean_mask

    tot = dict(tp=0, fp=0, fn=0, pred=0, truth=0, rp=0.0, rr=0.0,
               skel_truth=0, skel_pred=0, matched_truth=0, matched_pred=0)
    per_tile = []
    t0 = time.perf_counter()
    for i, pair in enumerate(pairs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")        # un-georeferenced PNGs
            rgb = _read_rgb(pair.image)
            truth = _read_mask(pair.label)
        h = min(rgb.shape[0], truth.shape[0])
        w = min(rgb.shape[1], truth.shape[1])
        rgb, truth = rgb[:h, :w], truth[:h, :w]
        pred = predict(rgb) > threshold

        tp = int((pred & truth).sum())
        fp = int((pred & ~truth).sum())
        fn = int((~pred & truth).sum())
        # relaxed: a prediction within slack of a label counts, and vice versa
        near_truth = ndimage.binary_dilation(truth, iterations=slack_px)
        near_pred = ndimage.binary_dilation(pred, iterations=slack_px)
        rp = float((pred & near_truth).sum())
        rr = float((truth & near_pred).sum())

        # centreline length, width-independent
        cleaned, _ = clean_mask(pred, min_object_px=100, min_hole_px=50, close_radius=2)
        sp = skeletonize(cleaned)
        st = skeletonize(truth)
        d_pred = ndimage.distance_transform_edt(~sp) if sp.any() else np.full(sp.shape, np.inf)
        d_truth = ndimage.distance_transform_edt(~st) if st.any() else np.full(st.shape, np.inf)
        mt = int((st & (d_pred <= tolerance_px)).sum())
        mp = int((sp & (d_truth <= tolerance_px)).sum())

        for k, v in dict(tp=tp, fp=fp, fn=fn, pred=int(pred.sum()), truth=int(truth.sum()),
                         rp=rp, rr=rr, skel_truth=int(st.sum()), skel_pred=int(sp.sum()),
                         matched_truth=mt, matched_pred=mp).items():
            tot[k] += v
        per_tile.append({"id": getattr(pair.image, "stem", str(pair.image)),
                         "iou": tp / max(tp + fp + fn, 1),
                         "completeness": mt / max(int(st.sum()), 1)})
        if progress:
            progress(i + 1, per_tile[-1]["id"])

    def safe(a, b):
        return a / b if b else 0.0

    p, r = safe(tot["tp"], tot["tp"] + tot["fp"]), safe(tot["tp"], tot["tp"] + tot["fn"])
    rp, rr = safe(tot["rp"], tot["pred"]), safe(tot["rr"], tot["truth"])
    comp = safe(tot["matched_truth"], tot["skel_truth"])
    corr = safe(tot["matched_pred"], tot["skel_pred"])
    return {
        "n_tiles": len(per_tile),
        "iou": round(safe(tot["tp"], tot["tp"] + tot["fp"] + tot["fn"]), 4),
        "f1": round(safe(2 * p * r, p + r), 4),
        "precision": round(p, 4),
        "recall": round(r, 4),
        "relaxed_f1": round(safe(2 * rp * rr, rp + rr), 4),
        "completeness": round(comp, 4),
        "correctness": round(corr, 4),
        "centreline_f1": round(safe(2 * comp * corr, comp + corr), 4),
        "seconds": round(time.perf_counter() - t0, 1),
        "per_tile": per_tile,
    }
