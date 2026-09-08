"""Segmentation backends.

Two models, one interface. ``sam3`` is zero-shot and takes a text prompt;
``unet`` is supervised on the Massachusetts Roads labels and ignores the
prompt. Both return a float confidence map, so everything downstream --
stitching, thresholding, vectorising, the MCP tools, the web app -- is
unchanged by the choice.
"""

from __future__ import annotations

BACKENDS = ("sam3", "unet")
DEFAULT_BACKEND = "sam3"


def make_segmenter(backend: str = DEFAULT_BACKEND, **kwargs):
    """Build a segmenter by name.

    Imports lazily: loading the SAM 3 stack costs seconds and several GB, and a
    caller asking for the U-Net should not pay for it.
    """
    backend = (backend or DEFAULT_BACKEND).lower()
    if backend == "sam3":
        from gisagent.segment.sam3 import Sam3RoadSegmenter

        return Sam3RoadSegmenter(**kwargs)
    if backend == "unet":
        from gisagent.segment.unet import UNetRoadSegmenter

        return UNetRoadSegmenter(**kwargs)
    raise ValueError(f"unknown segmentation backend {backend!r}; expected one of {BACKENDS}")
