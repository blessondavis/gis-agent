"""U-Net road segmenter trained on the Massachusetts Roads dataset.

Implements the same interface as :class:`~gisagent.segment.sam3.Sam3RoadSegmenter`
-- ``segment(path, prompt=..., threshold=..., upscale=...)`` returning a
:class:`SegmentResult` -- so it drops into the existing pipeline, MCP tools and
web app without any of them knowing which model produced the confidence map.

Differences from the SAM 3 backend, both inherent to a supervised model:

* ``prompt`` is accepted and ignored. There is no text conditioning; the model
  was trained on one class. It stays in the signature so the two backends stay
  interchangeable, and so a stored job manifest reads the same either way.
* ``n_instances`` is 1 when anything is predicted. This is semantic
  segmentation, not instance segmentation -- there are no separate objects to
  count.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from gisagent.config import get_settings
from gisagent.segment.sam3 import SegmentResult, _to_rgb_image

DEFAULT_CHECKPOINT = "models/unet_roads.pt"


class UNetRoadSegmenter:
    """Lazily-loaded supervised road segmenter."""

    def __init__(
        self,
        checkpoint: str | Path | None = None,
        *,
        device: str | None = None,
        tile: int = 512,
        overlap: int = 64,
    ) -> None:
        settings = get_settings()
        self.checkpoint = Path(checkpoint or DEFAULT_CHECKPOINT)
        self.device = device or settings.device
        self.tile = tile
        self.overlap = overlap
        self._model = None
        self._encoder = "resnet34"

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        if self._model is not None:
            return
        import segmentation_models_pytorch as smp
        import torch

        if not self.checkpoint.exists():
            raise FileNotFoundError(
                f"no trained checkpoint at {self.checkpoint}. "
                "Run `gisagent train` first."
            )
        blob = torch.load(self.checkpoint, map_location="cpu", weights_only=False)
        self._encoder = blob.get("encoder", "resnet34")
        model = smp.Unet(encoder_name=self._encoder, encoder_weights=None,
                         in_channels=3, classes=1)
        model.load_state_dict(blob["state_dict"])
        model.eval().to(self.device)
        self._model = model

    def unload(self) -> None:
        self._model = None

    # -- inference ---------------------------------------------------------- #

    def _predict(self, rgb: np.ndarray) -> np.ndarray:
        """Sliding-window inference over an HxWx3 uint8 array."""
        import torch

        h, w, _ = rgb.shape
        step = self.tile - self.overlap
        acc = np.zeros((h, w), dtype=np.float32)
        wgt = np.zeros((h, w), dtype=np.float32)

        # cosine taper so window seams do not show up as ridges in the output
        ramp = np.hanning(self.tile).astype(np.float32)
        window_w = np.clip(np.outer(ramp, ramp), 1e-3, None)

        rows = list(range(0, max(h - self.tile, 0) + 1, step)) or [0]
        cols = list(range(0, max(w - self.tile, 0) + 1, step)) or [0]
        if rows[-1] + self.tile < h:
            rows.append(h - self.tile)
        if cols[-1] + self.tile < w:
            cols.append(w - self.tile)

        with torch.no_grad():
            for r in rows:
                for c in cols:
                    patch = rgb[r:r + self.tile, c:c + self.tile]
                    ph, pw, _ = patch.shape
                    if ph < self.tile or pw < self.tile:
                        patch = np.pad(patch,
                                       ((0, self.tile - ph), (0, self.tile - pw),
                                        (0, 0)), mode="reflect")
                    x = torch.from_numpy(
                        np.ascontiguousarray(patch.transpose(2, 0, 1))
                    ).float().div_(255.0).unsqueeze(0).to(self.device)
                    with torch.amp.autocast("cuda", enabled=self.device == "cuda"):
                        prob = torch.sigmoid(self._model(x))
                    p = prob[0, 0].float().cpu().numpy()[:ph, :pw]
                    acc[r:r + ph, c:c + pw] += p * window_w[:ph, :pw]
                    wgt[r:r + ph, c:c + pw] += window_w[:ph, :pw]

        return acc / np.maximum(wgt, 1e-6)

    def segment(
        self,
        source,
        *,
        prompt: str = "road",
        threshold: float = 0.4,
        upscale: int = 1,
        **_ignored,
    ) -> SegmentResult:
        """Confidence map for one chip. ``prompt`` is accepted and ignored."""
        self.load()
        t0 = time.perf_counter()

        image = _to_rgb_image(source)
        rgb = np.asarray(image, dtype=np.uint8)
        h, w, _ = rgb.shape

        if upscale and upscale > 1:
            from PIL import Image

            big = image.resize((w * upscale, h * upscale), Image.BICUBIC)
            conf = self._predict(np.asarray(big, dtype=np.uint8))
            conf = np.asarray(
                Image.fromarray((conf * 255).astype(np.uint8)).resize(
                    (w, h), Image.BILINEAR
                ),
                dtype=np.float32,
            ) / 255.0
        else:
            conf = self._predict(rgb)

        peak = float(conf.max()) if conf.size else 0.0
        return SegmentResult(
            confidence=conf.astype(np.float32),
            n_instances=1 if peak >= threshold else 0,
            scores=[round(peak, 4)] if peak >= threshold else [],
            prompt=prompt,
            threshold=threshold,
            upscale=upscale,
            duration_s=time.perf_counter() - t0,
        )
