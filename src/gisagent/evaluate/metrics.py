"""Segmentation metrics for road extraction.

Roads cover roughly 0.5-5% of an aerial tile, so pixel accuracy is meaningless
here: predicting "no road" everywhere scores over 95%. IoU and F1 on the road
class are the numbers that mean something, and they are what this module
reports.

It also reports *relaxed* precision/recall. Road centrelines are annotated by
hand and are rarely pixel-exact, so the convention in the road-extraction
literature (Mnih & Hinton) is to count a prediction correct if it falls within a
few pixels of a true road. Strict IoU alone understates a prediction that traces
the right road one pixel to the left.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np


@dataclass
class Metrics:
    iou: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    relaxed_precision: float = 0.0
    relaxed_recall: float = 0.0
    relaxed_f1: float = 0.0
    slack_px: int = 3
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    pred_fraction: float = 0.0
    truth_fraction: float = 0.0
    n_pixels: int = 0

    def to_dict(self) -> dict:
        d = asdict(self)
        return {
            k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items()
        }

    def summary(self) -> str:
        return (
            f"IoU {self.iou:.3f} | F1 {self.f1:.3f} "
            f"(P {self.precision:.3f} R {self.recall:.3f}) | "
            f"relaxed F1 {self.relaxed_f1:.3f} "
            f"(P {self.relaxed_precision:.3f} R {self.relaxed_recall:.3f}) "
            f"@{self.slack_px}px"
        )


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask
    from skimage.morphology import dilation, disk

    return dilation(mask, disk(radius))


def compare_masks(
    prediction: np.ndarray,
    truth: np.ndarray,
    *,
    slack_px: int = 3,
    ignore: np.ndarray | None = None,
) -> Metrics:
    """Compare two boolean masks of identical shape."""
    pred = np.asarray(prediction).astype(bool)
    gt = np.asarray(truth).astype(bool)
    if pred.shape != gt.shape:
        raise ValueError(f"shape mismatch: prediction {pred.shape} vs truth {gt.shape}")

    if ignore is not None:
        keep = ~np.asarray(ignore).astype(bool)
        pred = pred & keep
        gt = gt & keep

    tp = int(np.logical_and(pred, gt).sum())
    fp = int(np.logical_and(pred, ~gt).sum())
    fn = int(np.logical_and(~pred, gt).sum())
    union = tp + fp + fn

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    # Relaxed: a predicted road pixel counts if truth is nearby, and vice versa.
    gt_near = _dilate(gt, slack_px)
    pred_near = _dilate(pred, slack_px)
    rp_den = int(pred.sum())
    rr_den = int(gt.sum())
    relaxed_precision = (
        int(np.logical_and(pred, gt_near).sum()) / rp_den if rp_den else 0.0
    )
    relaxed_recall = (
        int(np.logical_and(gt, pred_near).sum()) / rr_den if rr_den else 0.0
    )
    rf1 = (
        2 * relaxed_precision * relaxed_recall / (relaxed_precision + relaxed_recall)
        if (relaxed_precision + relaxed_recall)
        else 0.0
    )

    return Metrics(
        iou=(tp / union if union else 0.0),
        precision=precision,
        recall=recall,
        f1=f1,
        relaxed_precision=relaxed_precision,
        relaxed_recall=relaxed_recall,
        relaxed_f1=rf1,
        slack_px=slack_px,
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        pred_fraction=float(pred.mean()),
        truth_fraction=float(gt.mean()),
        n_pixels=int(pred.size),
    )


def _read_mask(path: Path | str) -> np.ndarray:
    import rasterio

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with rasterio.open(path) as ds:
            arr = ds.read(1)
    return arr > (127 if arr.dtype == np.uint8 else 0.5)


def evaluate_rasters(
    prediction_path: Path | str,
    truth_path: Path | str,
    *,
    slack_px: int = 3,
) -> Metrics:
    """Compare two mask rasters on disk."""
    return compare_masks(
        _read_mask(prediction_path), _read_mask(truth_path), slack_px=slack_px
    )


def sweep_threshold(
    confidence: np.ndarray,
    truth: np.ndarray,
    *,
    thresholds: list[float] | None = None,
    slack_px: int = 3,
) -> list[dict]:
    """Score a confidence map at several cut-offs.

    This is the feedback the agent uses to pick an operating point instead of
    accepting whatever the model's default threshold happens to give.
    """
    if thresholds is None:
        thresholds = [round(0.05 * i, 2) for i in range(1, 20)]
    gt = np.asarray(truth).astype(bool)
    rows = []
    for t in thresholds:
        m = compare_masks(np.asarray(confidence) >= t, gt, slack_px=slack_px)
        rows.append({"threshold": t, **m.to_dict()})
    return rows


def best_threshold(rows: list[dict], key: str = "f1") -> dict:
    return max(rows, key=lambda r: r.get(key, 0.0)) if rows else {}
