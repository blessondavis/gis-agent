"""SAM 3 road segmentation.

SAM 3 does Promptable Concept Segmentation: given a short text phrase it returns
instance masks for everything matching that concept. That is a much better fit
for roads than SAM 1/2, whose automatic mask generator produces class-agnostic
blobs that then have to be guessed at.

Two knobs matter for aerial imagery and both are exposed so the agent can tune
them per chip:

* ``prompt`` - "road" is not always the best phrase. Concept models are
  sensitive to wording, and the right phrase varies with what is on the ground.
* ``upscale`` - at 1 m/px a residential road is only ~8 px wide. Upsampling the
  chip before inference makes roads occupy a scale the model actually saw during
  training, at the cost of GPU memory.

Output is a float confidence map rather than a hard mask, so the stitcher can
blend overlaps and the threshold stays a downstream decision.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from gisagent.config import get_settings

DEFAULT_PROMPT = "road"


@dataclass
class SegmentResult:
    """Result of segmenting one chip."""

    confidence: np.ndarray                 # float32 HxW in [0, 1]
    n_instances: int = 0
    scores: list[float] = field(default_factory=list)
    prompt: str = DEFAULT_PROMPT
    threshold: float = 0.4
    upscale: int = 1
    duration_s: float = 0.0

    @property
    def coverage(self) -> float:
        return float((self.confidence > 0).mean())

    def to_dict(self) -> dict:
        return {
            "n_instances": self.n_instances,
            "scores": [round(s, 4) for s in self.scores[:20]],
            "max_score": round(max(self.scores), 4) if self.scores else 0.0,
            "coverage": round(self.coverage, 5),
            "prompt": self.prompt,
            "threshold": self.threshold,
            "upscale": self.upscale,
            "duration_s": round(self.duration_s, 2),
        }


def _to_rgb_image(source):
    """Accept a path, HxWx3/3xHxW array, or PIL image; return a PIL RGB image."""
    from PIL import Image

    if isinstance(source, Image.Image):
        return source.convert("RGB")

    if isinstance(source, (str, Path)):
        import rasterio

        with rasterio.open(source) as ds:
            bands = min(3, ds.count)
            arr = ds.read(list(range(1, bands + 1)))
        if arr.shape[0] == 1:
            arr = np.repeat(arr, 3, axis=0)
        arr = np.transpose(arr, (1, 2, 0))
    else:
        arr = np.asarray(source)
        if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[0] < arr.shape[-1]:
            arr = np.transpose(arr, (1, 2, 0))
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)

    arr = arr[:, :, :3]
    if arr.dtype != np.uint8:
        finite = arr[np.isfinite(arr)]
        hi = float(finite.max()) if finite.size else 1.0
        arr = (arr / hi * 255.0) if hi > 0 else arr
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


class Sam3RoadSegmenter:
    """Lazily-loaded SAM 3 wrapper specialised for linear features."""

    def __init__(
        self,
        model_id: str | None = None,
        *,
        device: str | None = None,
        dtype: str = "float16",
        token: str | None = None,
    ) -> None:
        settings = get_settings()
        self.model_id = model_id or settings.sam_model
        self.device = device or settings.device
        self.dtype = dtype
        self.token = token or settings.hf_token
        self._model = None
        self._processor = None

    # -- lifecycle ---------------------------------------------------------- #

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import Sam3Model, Sam3Processor

        torch_dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }.get(self.dtype, torch.float16)
        if self.device == "cpu":
            torch_dtype = torch.float32  # fp16 on CPU is slower, not faster

        self._processor = Sam3Processor.from_pretrained(self.model_id, token=self.token)
        model = Sam3Model.from_pretrained(
            self.model_id, token=self.token, dtype=torch_dtype
        )
        self._model = model.to(self.device).eval()

    def unload(self) -> None:
        self._model = None
        self._processor = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    # -- inference ---------------------------------------------------------- #

    def segment(
        self,
        source,
        *,
        prompt: str = DEFAULT_PROMPT,
        threshold: float = 0.4,
        mask_threshold: float = 0.5,
        upscale: int = 1,
        max_instances: int | None = None,
    ) -> SegmentResult:
        """Segment one chip and return a float confidence map."""
        import time

        import torch

        self.load()
        image = _to_rgb_image(source)
        native_size = (image.height, image.width)

        if upscale and upscale > 1:
            from PIL import Image

            image = image.resize(
                (image.width * upscale, image.height * upscale), Image.BICUBIC
            )

        t0 = time.perf_counter()
        inputs = self._processor(images=image, text=prompt, return_tensors="pt")
        inputs = inputs.to(self.device)

        with torch.inference_mode():
            outputs = self._model(**inputs)

        target = [[image.height, image.width]]
        results = self._processor.post_process_instance_segmentation(
            outputs,
            threshold=threshold,
            mask_threshold=mask_threshold,
            target_sizes=target,
        )[0]
        duration = time.perf_counter() - t0

        masks = results.get("masks")
        scores = results.get("scores")
        scores = [] if scores is None else [float(s) for s in np.atleast_1d(
            scores.detach().cpu().numpy() if hasattr(scores, "detach") else scores
        )]

        conf = np.zeros((image.height, image.width), dtype=np.float32)
        n = 0
        if masks is not None and len(masks) > 0:
            arr = masks.detach().cpu().numpy() if hasattr(masks, "detach") else np.asarray(masks)
            if arr.ndim == 2:
                arr = arr[None]
            order = np.argsort(scores)[::-1] if len(scores) == len(arr) else range(len(arr))
            for rank, idx in enumerate(order):
                if max_instances is not None and rank >= max_instances:
                    break
                m = arr[idx] > 0.5
                if not m.any():
                    continue
                s = float(scores[idx]) if idx < len(scores) else 1.0
                # keep the strongest claim on each pixel
                np.maximum(conf, m.astype(np.float32) * s, out=conf)
                n += 1

        if upscale and upscale > 1:
            from PIL import Image

            conf = np.asarray(
                Image.fromarray(conf).resize(
                    (native_size[1], native_size[0]), Image.BILINEAR
                ),
                dtype=np.float32,
            )

        return SegmentResult(
            confidence=conf,
            n_instances=n,
            scores=scores,
            prompt=prompt,
            threshold=threshold,
            upscale=upscale,
            duration_s=duration,
        )

    def segment_chips(
        self,
        chips: list,
        *,
        prompt: str = DEFAULT_PROMPT,
        threshold: float = 0.4,
        upscale: int = 1,
        progress=None,
    ) -> dict[str, SegmentResult]:
        """Segment a list of ChipSpec, returning {chip_id: SegmentResult}."""
        out: dict[str, SegmentResult] = {}
        for i, spec in enumerate(chips):
            src = spec.path
            if src is None:
                raise ValueError(
                    f"chip {spec.chip_id} has no written file; "
                    "tile with write_chips=True"
                )
            out[spec.chip_id] = self.segment(
                src, prompt=prompt, threshold=threshold, upscale=upscale
            )
            if progress is not None:
                progress(i + 1, len(chips), spec.chip_id, out[spec.chip_id])
        return out
