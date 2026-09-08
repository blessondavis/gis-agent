"""Vision-model critic: judge an annotation from the picture alone.

This is what makes unlabelled imagery workable. With ground truth you can
measure IoU and stop when it is good enough; without it, something still has to
say whether the annotation is finished. A vision model looking at the overlay
can, and it was measured before being built on: Spearman rho = 0.886 against
six degradations of known quality (docs/vlm-critic.md).

Two things that document records and this module encodes:

* **It is blind to misregistration.** A uniformly shifted annotation still
  looks like roads drawn over roads, and scored a middling 50 despite an IoU of
  0.127. Never use this score as the only stop condition -- pair it with the
  topology checks in :mod:`gisagent.vector.topology`, which catch exactly that.
* **Payload size decides whether it answers at all.** A 640 px JPEG at quality
  85 is ~259 KB of base64 and times out every time; 512 px at quality 72 is
  ~115 KB and answers in about 30 s.

The annotation is drawn in magenta because magenta essentially never occurs in
aerial photography, so the critic cannot mistake the overlay for terrain.
"""

from __future__ import annotations

import base64
import io
import json
import time
from dataclasses import dataclass, field, asdict

import numpy as np

from gisagent.config import get_settings

OVERLAY_RGB = (255.0, 0.0, 200.0)      # magenta
OVERLAY_ALPHA = 0.75
TRANSIENT = ("429", "503", "ResourceExhausted", "timed out", "Timeout")

SYSTEM = (
    "You review road annotations on aerial imagery. You see an aerial photo "
    "with a road annotation drawn over it in bright magenta. Judge only how "
    "well the magenta covers the real roads.\n"
    "Reply with strict JSON and nothing else:\n"
    '{"completeness": 0-100, "correctness": 0-100, "overall": 0-100, '
    '"verdict": "good"|"usable"|"poor"|"failed", '
    '"problems": ["short phrase", ...], "advice": "one sentence"}\n'
    "completeness = percentage of the real roads that are marked magenta. "
    "correctness = percentage of the magenta that really is road. "
    "If whole streets are unmarked, completeness must be low. "
    "Be strict and numeric."
)


@dataclass
class Critique:
    """What the critic thought of one view of an annotation."""

    completeness: float = -1.0
    correctness: float = -1.0
    overall: float = -1.0
    verdict: str = ""
    problems: list[str] = field(default_factory=list)
    advice: str = ""
    model: str = ""
    window: tuple[int, int, int, int] | None = None
    duration_s: float = 0.0
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error and self.overall >= 0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["window"] = list(self.window) if self.window else None
        d["ok"] = self.ok
        return d


def render_overlay(
    rgb: np.ndarray, mask: np.ndarray, *, alpha: float = OVERLAY_ALPHA
) -> "Image.Image":                                          # noqa: F821
    """Blend a boolean mask over an HxWx3 uint8 image in magenta."""
    from PIL import Image

    if rgb.ndim != 3 or rgb.shape[2] < 3:
        raise ValueError(f"expected HxWx3 imagery, got {rgb.shape}")
    h = min(rgb.shape[0], mask.shape[0])
    w = min(rgb.shape[1], mask.shape[1])
    base = rgb[:h, :w, :3].astype(np.float32)
    m = mask[:h, :w].astype(bool)[..., None]
    colour = np.array(OVERLAY_RGB, dtype=np.float32)
    blended = np.where(m, (1.0 - alpha) * base + alpha * colour, base)
    return Image.fromarray(blended.clip(0, 255).astype(np.uint8))


def _encode(image, px: int, quality: int) -> str:
    from PIL import Image

    if image.size != (px, px):
        image = image.resize((px, px), Image.LANCZOS)
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


class RoadCritic:
    """Asks a vision model how good an annotation looks."""

    def __init__(
        self,
        model: str | None = None,
        *,
        image_px: int | None = None,
        jpeg_quality: int | None = None,
        timeout: float = 180.0,
    ) -> None:
        s = get_settings()
        self.model = model or s.vlm_model
        self.image_px = image_px or s.vlm_image_px
        self.jpeg_quality = jpeg_quality or s.vlm_jpeg_quality
        self.timeout = timeout
        self._client = None

    def client(self):
        if self._client is None:
            from openai import OpenAI

            s = get_settings()
            if not s.openai_api_key:
                raise RuntimeError("OPENAI_API_KEY is not set")
            # retries handled here so backoff can be reported, not silent
            self._client = OpenAI(
                api_key=s.openai_api_key, base_url=s.openai_base_url,
                timeout=self.timeout, max_retries=0,
            )
        return self._client

    def review_image(
        self, image, *, question: str = "", attempts: int = 4, progress=None
    ) -> Critique:
        """Critique a already-rendered overlay image."""
        t0 = time.perf_counter()
        payload = _encode(image, self.image_px, self.jpeg_quality)
        prompt = question or "Review this road annotation. JSON only."

        last = ""
        for i in range(attempts):
            try:
                resp = self.client().chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": SYSTEM},
                        {"role": "user", "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {
                                "url": f"data:image/jpeg;base64,{payload}"}},
                        ]},
                    ],
                    max_tokens=900,
                    temperature=0.1,
                )
                text = (resp.choices[0].message.content or "").strip()
                return self._parse(text, time.perf_counter() - t0)
            except Exception as exc:                       # noqa: BLE001
                last = str(exc)
                if any(t in last for t in TRANSIENT) and i < attempts - 1:
                    wait = 15 * (i + 1)
                    if progress:
                        progress(f"critic busy, retrying in {wait}s")
                    time.sleep(wait)
                    continue
                break

        return Critique(model=self.model, error=last[:300],
                        duration_s=time.perf_counter() - t0)

    def review(
        self, rgb: np.ndarray, mask: np.ndarray, *,
        window: tuple[int, int, int, int] | None = None, **kw
    ) -> Critique:
        """Critique a mask over imagery."""
        c = self.review_image(render_overlay(rgb, mask), **kw)
        c.window = window
        return c

    def _parse(self, text: str, duration: float) -> Critique:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            return Critique(model=self.model, duration_s=duration,
                            error=f"no JSON in reply: {text[:160]}")
        try:
            d = json.loads(text[start:end + 1])
        except Exception as exc:                            # noqa: BLE001
            return Critique(model=self.model, duration_s=duration,
                            error=f"bad JSON: {exc}")

        def num(key: str) -> float:
            v = d.get(key, -1)
            try:
                return float(v)
            except (TypeError, ValueError):
                return -1.0

        problems = d.get("problems") or []
        if isinstance(problems, str):
            problems = [problems]
        return Critique(
            completeness=num("completeness"),
            correctness=num("correctness"),
            overall=num("overall"),
            verdict=str(d.get("verdict", "")),
            problems=[str(p)[:160] for p in problems][:8],
            advice=str(d.get("advice", ""))[:300],
            model=self.model,
            duration_s=duration,
        )


def available() -> bool:
    return bool(get_settings().openai_api_key)
