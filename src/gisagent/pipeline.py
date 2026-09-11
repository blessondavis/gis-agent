"""The road-annotation pipeline, as a set of resumable operations on a job.

Every stage writes its artefact to the job directory and records what it did in
``job.json``. Three consumers share this one implementation:

* the MCP server, which exposes these operations as tools for the agent,
* the FastAPI app, which drives them directly for non-agent runs,
* the CLI, for reproducing a run without a browser.

Stages are independent and re-runnable, which is what lets the agent change a
prompt or a threshold and redo only the affected step instead of starting over.
"""

from __future__ import annotations

import json
import time
import uuid
import warnings
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio

from gisagent.config import get_settings
from gisagent.raster.tiling import ChipSpec, load_chips, tile_raster, stitch_masks

STAGES = ("created", "tiled", "segmented", "stitched", "vectorized", "evaluated")


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class StageRecord:
    stage: str
    at: str = field(default_factory=_utc)
    params: dict = field(default_factory=dict)
    result: dict = field(default_factory=dict)
    duration_s: float = 0.0


class Job:
    """A single annotation run, backed by a directory on disk."""

    def __init__(self, job_dir: Path | str) -> None:
        self.dir = Path(job_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.dir / "job.json"
        self.manifest: dict = (
            json.loads(self.manifest_path.read_text(encoding="utf-8"))
            if self.manifest_path.exists()
            else {}
        )

    # -- paths -------------------------------------------------------------- #

    @property
    def job_id(self) -> str:
        return self.manifest.get("job_id", self.dir.name)

    @property
    def image_path(self) -> Path:
        return self.dir / "mosaic.tif"

    @property
    def truth_path(self) -> Path:
        return self.dir / "truth.tif"

    @property
    def chips_dir(self) -> Path:
        return self.dir / "chips"

    @property
    def conf_dir(self) -> Path:
        return self.dir / "conf"

    @property
    def conf_path(self) -> Path:
        return self.dir / "confidence.tif"

    @property
    def mask_path(self) -> Path:
        return self.dir / "mask.tif"

    @property
    def vector_path(self) -> Path:
        return self.dir / "roads.geojson"

    @property
    def edits_path(self) -> Path:
        return self.dir / "edits.json"

    @property
    def network_path(self) -> Path:
        """The deliverable: machine centrelines with the person's edits applied."""
        return self.dir / "network.geojson"

    @property
    def suggestions_path(self) -> Path:
        return self.dir / "suggestions.geojson"

    @property
    def preview_dir(self) -> Path:
        return self.dir / "preview"

    # -- state -------------------------------------------------------------- #

    def save(self) -> None:
        self.manifest_path.write_text(
            json.dumps(self.manifest, indent=2, default=str), encoding="utf-8"
        )

    def record(self, stage: str, params: dict, result: dict, duration: float) -> None:
        self.manifest.setdefault("stages", []).append(
            asdict(StageRecord(stage=stage, params=params, result=result,
                               duration_s=round(duration, 2)))
        )
        self.manifest["stage"] = stage
        self.manifest["updated_at"] = _utc()
        self.save()

    def has_truth(self) -> bool:
        return self.truth_path.exists()

    def _preserve(self, path: Path) -> None:
        """Let an active checkpoint keep the old copy before it is overwritten."""
        from gisagent.checkpoints import Checkpoints

        Checkpoints(self.dir).preserve(path)

    def status(self) -> dict:
        return {
            "job_id": self.job_id,
            "stage": self.manifest.get("stage", "created"),
            "region": self.manifest.get("region", {}),
            "artifacts": {
                "image": self.image_path.exists(),
                "truth": self.truth_path.exists(),
                "chips": (self.chips_dir / "chips.json").exists(),
                "confidence": self.conf_path.exists(),
                "mask": self.mask_path.exists(),
                "vectors": self.vector_path.exists(),
            },
            "metrics": self.manifest.get("metrics"),
            "vector_stats": self.manifest.get("vector_stats"),
            "n_stages": len(self.manifest.get("stages", [])),
            "updated_at": self.manifest.get("updated_at"),
        }

    # -- stage 1: build the region ----------------------------------------- #

    def build_region(self, sat_tiles: list[Path], truth_tiles: list[Path] | None) -> dict:
        """Mosaic the input tiles (and matching ground truth) into the job."""
        from rasterio.merge import merge

        t0 = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            srcs = [rasterio.open(p) for p in sat_tiles]
            try:
                mosaic, transform = merge(srcs)
                profile = srcs[0].profile.copy()
                crs = srcs[0].crs
            finally:
                for s in srcs:
                    s.close()

        profile.update(
            driver="GTiff", height=mosaic.shape[1], width=mosaic.shape[2],
            count=mosaic.shape[0], transform=transform, crs=crs,
            compress="deflate", tiled=True, blockxsize=256, blockysize=256,
        )
        with rasterio.open(self.image_path, "w", **profile) as dst:
            dst.write(mosaic)

        result = {
            "width": int(mosaic.shape[2]),
            "height": int(mosaic.shape[1]),
            "bands": int(mosaic.shape[0]),
            "crs": str(crs),
            "n_tiles": len(sat_tiles),
        }

        if truth_tiles:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                tsrcs = [rasterio.open(p) for p in truth_tiles]
                try:
                    tmos, ttf = merge(tsrcs)
                finally:
                    for s in tsrcs:
                        s.close()
            tprof = profile.copy()
            tprof.update(count=1, dtype="uint8", transform=ttf,
                         height=tmos.shape[1], width=tmos.shape[2])
            tprof.pop("photometric", None)
            with rasterio.open(self.truth_path, "w", **tprof) as dst:
                dst.write((tmos[0] > 127).astype("uint8") * 255, 1)
            result["truth"] = True
            result["truth_fraction"] = round(float((tmos[0] > 127).mean()), 6)

        # geographic extent, for the map UI
        result["bounds_wgs84"] = self.bounds_wgs84()
        self.manifest["region"] = result
        self.record("created", {"n_tiles": len(sat_tiles)}, result,
                    time.perf_counter() - t0)
        return result

    def bounds_wgs84(self) -> list[float]:
        from rasterio.warp import transform_bounds

        with rasterio.open(self.image_path) as ds:
            return [round(v, 6) for v in
                    transform_bounds(ds.crs, "EPSG:4326", *ds.bounds)]

    # -- stage 2: tiling ---------------------------------------------------- #

    def tile(self, chip_size: int = 1024, overlap: int = 128) -> dict:
        t0 = time.perf_counter()
        specs = tile_raster(self.image_path, self.chips_dir,
                            chip_size=chip_size, overlap=overlap)
        result = {
            "n_chips": len(specs),
            "chip_size": chip_size,
            "overlap": overlap,
            "chip_ids": [s.chip_id for s in specs],
        }
        self.manifest["tiling"] = {"chip_size": chip_size, "overlap": overlap,
                                   "n_chips": len(specs)}
        self.record("tiled", {"chip_size": chip_size, "overlap": overlap},
                    result, time.perf_counter() - t0)
        return result

    def chips(self) -> list[ChipSpec]:
        return load_chips(self.chips_dir)

    # -- stage 3: segmentation --------------------------------------------- #

    def segment(
        self,
        segmenter,
        *,
        prompt: str = "road",
        threshold: float = 0.4,
        upscale: int = 1,
        chip_ids: list[str] | None = None,
        progress=None,
    ) -> dict:
        t0 = time.perf_counter()
        self.conf_dir.mkdir(parents=True, exist_ok=True)
        specs = self.chips()
        if chip_ids:
            wanted = set(chip_ids)
            specs = [s for s in specs if s.chip_id in wanted]

        per_chip = {}
        for i, spec in enumerate(specs):
            res = segmenter.segment(spec.path, prompt=prompt,
                                    threshold=threshold, upscale=upscale)
            self._preserve(self.conf_dir / f"{spec.chip_id}.npy")
            np.save(self.conf_dir / f"{spec.chip_id}.npy",
                    res.confidence.astype(np.float32))
            per_chip[spec.chip_id] = res.to_dict()
            if progress is not None:
                progress(i + 1, len(specs), spec.chip_id, res)

        result = {
            "n_chips": len(specs),
            "prompt": prompt,
            "threshold": threshold,
            "upscale": upscale,
            "per_chip": per_chip,
            "mean_coverage": round(
                float(np.mean([c["coverage"] for c in per_chip.values()])), 5
            ) if per_chip else 0.0,
            "total_instances": int(sum(c["n_instances"] for c in per_chip.values())),
        }
        self.manifest["segmentation"] = {
            k: result[k] for k in
            ("prompt", "threshold", "upscale", "mean_coverage", "total_instances")
        }
        self.record("segmented",
                    {"prompt": prompt, "threshold": threshold, "upscale": upscale},
                    {k: v for k, v in result.items() if k != "per_chip"},
                    time.perf_counter() - t0)
        return result

    def load_confidence(self) -> dict[str, np.ndarray]:
        return {
            p.stem: np.load(p) for p in sorted(self.conf_dir.glob("*.npy"))
        }

    # -- stage 4: stitching ------------------------------------------------- #

    def stitch(self, threshold: float = 0.5) -> dict:
        t0 = time.perf_counter()
        specs = self.chips()
        conf = self.load_confidence()
        if not conf:
            raise RuntimeError("no chip confidences found; run segmentation first")

        overlap = self.manifest.get("tiling", {}).get("overlap", 128)
        self._preserve(self.conf_path)
        stitch_masks(conf, specs, self.image_path, self.conf_path,
                     overlap=overlap, threshold=None)
        stitch_masks(conf, specs, self.image_path, self.mask_path,
                     overlap=overlap, threshold=threshold)

        with rasterio.open(self.mask_path) as ds:
            mask = ds.read(1) > 127
        result = {
            "threshold": threshold,
            "mask_fraction": round(float(mask.mean()), 6),
            "mask_pixels": int(mask.sum()),
        }
        self.record("stitched", {"threshold": threshold}, result,
                    time.perf_counter() - t0)
        return result

    # -- stage 5: vectorize ------------------------------------------------- #

    # -- targeted rework ---------------------------------------------------- #

    def pixel_window_from_wgs84(self, bbox: list[float]) -> tuple[int, int, int, int]:
        """(west, south, east, north) in WGS84 -> (col, row, width, height)."""
        from rasterio.warp import transform_bounds

        with rasterio.open(self.image_path) as ds:
            west, south, east, north = transform_bounds("EPSG:4326", ds.crs, *bbox)
            inv = ~ds.transform
            c0, r0 = inv * (west, north)
            c1, r1 = inv * (east, south)
            col0, col1 = sorted((int(c0), int(c1)))
            row0, row1 = sorted((int(r0), int(r1)))
            col0 = max(0, min(col0, ds.width - 1))
            row0 = max(0, min(row0, ds.height - 1))
            col1 = max(col0 + 1, min(col1, ds.width))
            row1 = max(row0 + 1, min(row1, ds.height))
        return col0, row0, col1 - col0, row1 - row0

    def chips_in_window(self, window: tuple[int, int, int, int]) -> list[ChipSpec]:
        """Chips overlapping a pixel window, which is the unit of rework."""
        col, row, w, h = window
        hits = []
        for s in self.chips():
            if (s.col_off < col + w and s.col_off + s.width > col
                    and s.row_off < row + h and s.row_off + s.height > row):
                hits.append(s)
        return hits

    def refine(
        self,
        segmenter,
        *,
        bbox_wgs84: list[float] | None = None,
        window: tuple[int, int, int, int] | None = None,
        prompt: str = "road network",
        threshold: float = 0.3,
        upscale: int = 2,
        note: str = "",
        progress=None,
    ) -> dict:
        """Re-run segmentation over one area only, then rebuild the outputs.

        This is what makes a correction cheap: a user pointing at a bad corner
        should not cost a full re-run of the region. Only the chips touching
        that area are re-inferred; the rest of the confidence maps are reused,
        and stitching recombines them into a consistent whole.
        """
        t0 = time.perf_counter()
        if window is None:
            if not bbox_wgs84:
                raise ValueError("pass either bbox_wgs84 or window")
            window = self.pixel_window_from_wgs84(bbox_wgs84)

        targets = self.chips_in_window(window)
        if not targets:
            raise RuntimeError("no chips overlap that area")

        self.conf_dir.mkdir(parents=True, exist_ok=True)
        for i, spec in enumerate(targets):
            res = segmenter.segment(spec.path, prompt=prompt,
                                    threshold=threshold, upscale=upscale)
            self._preserve(self.conf_dir / f"{spec.chip_id}.npy")
            np.save(self.conf_dir / f"{spec.chip_id}.npy",
                    res.confidence.astype(np.float32))
            if progress is not None:
                progress(i + 1, len(targets), spec.chip_id, res)

        result = {
            "window": list(window),
            "bbox_wgs84": bbox_wgs84,
            "chips_reworked": [s.chip_id for s in targets],
            "n_chips": len(targets),
            "prompt": prompt,
            "threshold": threshold,
            "upscale": upscale,
            "note": note,
        }
        self.manifest.setdefault("refinements", []).append(
            {**result, "at": _utc()}
        )
        self.record("segmented", {"refine": True, "prompt": prompt}, result,
                    time.perf_counter() - t0)
        return result

    def vectorize(self, **kwargs) -> dict:
        from gisagent.vector.roads import vectorize_roads

        t0 = time.perf_counter()
        kwargs.setdefault("to_wgs84", True)
        kwargs.setdefault("confidence_path", self.conf_path)
        _, stats = vectorize_roads(self.mask_path, self.vector_path, **kwargs)
        result = stats.to_dict()
        self.manifest["vector_stats"] = result
        self.record("vectorized", {k: v for k, v in kwargs.items()}, result,
                    time.perf_counter() - t0)
        # the machine layer changed underneath any human edits; replay them
        self.rebuild_network()
        return result

    # -- human edits -------------------------------------------------------- #

    def edit_log(self):
        from gisagent.vector.edits import EditLog

        return EditLog(self.edits_path)

    def rebuild_network(self) -> dict:
        """Merge the edit log over the machine centrelines into network.geojson.

        Deliberately writes nothing to job.json: the web process (edits) and
        the MCP process (vectorising) both call this, and the manifest is the
        one file they would otherwise race on. Stats ride in the GeoJSON.
        """
        from gisagent.vector.edits import merge_network

        machine = (json.loads(self.vector_path.read_text(encoding="utf-8"))
                   if self.vector_path.exists() else None)
        fc, _ = merge_network(machine, self.edit_log().state(),
                              crs_hint=(self.manifest.get("region") or {}).get("crs"))
        tmp = self.network_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(fc), encoding="utf-8")
        tmp.replace(self.network_path)
        return fc

    def network(self) -> dict:
        if not self.network_path.exists() or (
            self.vector_path.exists()
            and self.vector_path.stat().st_mtime > self.network_path.stat().st_mtime
        ):
            return self.rebuild_network()
        return json.loads(self.network_path.read_text(encoding="utf-8"))

    def suggest(self, threshold: float = 0.25, limit: int = 60) -> dict:
        from gisagent.vector.edits import suggest_roads

        if not self.conf_path.exists():
            raise RuntimeError("no confidence map yet - segment and stitch first")
        fc = suggest_roads(self.conf_path, self.network(),
                           self.edit_log().state().dismissed,
                           threshold=threshold, limit=limit)
        self.suggestions_path.write_text(json.dumps(fc), encoding="utf-8")
        return fc

    # -- candidates: several settings side by side, then apply one --------- #

    VECTOR_KEYS = ("min_object_px", "min_hole_px", "close_radius",
                   "simplify_tolerance_m", "min_length_m")

    # (type, min, max) per setting. Variants come from a model writing JSON,
    # so they are validated rather than trusted: a live run sent
    # "mask_threshold" (silently ignored -> five identical candidates) and
    # close_radius 0.5 (an empty morphology footprint -> a crash).
    VARIANT_SPEC = {
        "threshold": (float, 0.01, 0.99),
        "min_object_px": (int, 0, 100_000),
        "min_hole_px": (int, 0, 100_000),
        "close_radius": (int, 0, 10),
        "simplify_tolerance_m": (float, 0.0, 20.0),
        "min_length_m": (float, 0.0, 500.0),
    }
    VARIANT_ALIASES = {"mask_threshold": "threshold", "thr": "threshold"}

    @classmethod
    def normalise_variant(cls, variant: dict) -> dict:
        if not isinstance(variant, dict):
            raise ValueError(f"a variant must be an object, got {variant!r}")
        out = {}
        for key, value in variant.items():
            key = cls.VARIANT_ALIASES.get(key, key)
            if key not in cls.VARIANT_SPEC:
                raise ValueError(f"unknown setting {key!r}; allowed: "
                                 f"{', '.join(cls.VARIANT_SPEC)}")
            typ, lo, hi = cls.VARIANT_SPEC[key]
            try:
                x = int(round(float(value))) if typ is int else float(value)
            except (TypeError, ValueError):
                raise ValueError(f"{key} must be a number, got {value!r}") from None
            if not lo <= x <= hi:
                raise ValueError(f"{key}={value} is outside [{lo}, {hi}]")
            out[key] = x
        out.setdefault("threshold", 0.5)
        return out

    def try_candidates(self, variants: list[dict], *, objective: str = "balanced",
                       slack_px: int = 3, workers: int = 3) -> dict:
        """Build and score each variant in isolation; change nothing live.

        Each variant is a mask threshold plus optional vectoriser settings,
        applied to the *existing* confidence map -- no re-inference, so a
        handful of candidates costs seconds, not minutes. Every candidate gets
        its own directory (the worktree idea: isolated state, explicit apply)
        and the person's edits are merged over each, so the ranking is of the
        network they would actually get.

        Judge: an F-beta of precision and recall of the network *by length*,
        with beta set by ``objective`` -- "balanced" (1), "precision" (0.5:
        a wrong road costs more than a missing one) or "recall" (2). With
        ground truth those are correctness and completeness; without it, their
        expected values under the model's own confidence.

        The label-free judge was measured, not assumed. On Boston, over seven
        variants, it picked the ground-truth winner under all three objectives
        (Spearman 0.89 / 0.96 / 1.00), where topology ranked the same
        candidates backwards (-0.64). Topology is still reported, as a
        diagnostic.
        """
        from gisagent.evaluate.network import BETA, expected_scores, f_beta

        if objective not in BETA:
            raise ValueError(f"objective must be one of {list(BETA)}")
        beta = BETA[objective]
        import shutil
        from concurrent.futures import ThreadPoolExecutor

        from gisagent.evaluate.metrics import compare_masks
        from gisagent.evaluate.network import score_network
        from gisagent.vector.edits import merge_network
        from gisagent.vector.roads import vectorize_roads
        from gisagent.vector.topology import analyse

        if not variants:
            raise ValueError("give at least one variant")
        # validate everything before spending any time; drop exact repeats
        unique: dict[str, dict] = {}
        for v in variants:
            n = self.normalise_variant(v)
            unique.setdefault(json.dumps(n, sort_keys=True), n)
        variants = list(unique.values())
        if not self.conf_path.exists():
            raise RuntimeError("no stitched confidence map yet - run stitch_result first")
        with rasterio.open(self.conf_path) as ds:
            conf = ds.read(1).astype(np.float32)
            profile = ds.profile.copy()
            conf_tf, conf_crs = ds.transform, ds.crs
        profile.update(dtype="uint8", count=1, nodata=None)
        truth = None
        if self.has_truth():
            with rasterio.open(self.truth_path) as ds:
                truth = ds.read(1) > 127

        root = self.dir / "candidates"
        shutil.rmtree(root, ignore_errors=True)
        root.mkdir(parents=True)
        edits = self.edit_log().state()
        crs_hint = (self.manifest.get("region") or {}).get("crs")

        def build(i: int, v: dict) -> dict:
            cid = f"c{i + 1}"
            d = root / cid
            d.mkdir()
            thr = v["threshold"]
            vec_kw = {k: v[k] for k in self.VECTOR_KEYS if k in v}
            mask = conf > thr
            with rasterio.open(d / "mask.tif", "w", **profile) as dst:
                dst.write(mask.astype("uint8") * 255, 1)
            vectorize_roads(d / "mask.tif", d / "roads.geojson", to_wgs84=True,
                            confidence_path=self.conf_path, **vec_kw)
            machine = json.loads((d / "roads.geojson").read_text(encoding="utf-8"))
            net, nstats = merge_network(machine, edits, crs_hint=crs_hint)
            topo = analyse(net)
            exp = expected_scores(net, conf, conf_tf, conf_crs)
            row = {"id": cid, "params": {"threshold": thr, **vec_kw},
                   "n_features": nstats.n_features,
                   "length_km": round(nstats.total_length_m / 1000, 2),
                   "expected_precision": round(exp["precision"], 4),
                   "expected_recall": round(exp["recall"], 4),
                   "expected_f1": round(exp["f1"], 4),
                   "topology_score": round(topo.score, 4),
                   "components": topo.n_components, "dangles": topo.n_dangles}
            if truth is not None:
                m = compare_masks(mask, truth, slack_px=slack_px).to_dict()
                s = score_network(net, self.truth_path)["overall"]
                row.update({k: m[k] for k in ("iou", "f1", "relaxed_f1")},
                           completeness=s["completeness"],
                           correctness=s["correctness"], quality=s["quality"])
                row["score"] = round(f_beta(s["correctness"], s["completeness"], beta), 4)
            else:
                row["score"] = round(f_beta(exp["precision"], exp["recall"], beta), 4)
            return row

        with ThreadPoolExecutor(max_workers=max(1, min(workers, len(variants)))) as ex:
            rows = list(ex.map(lambda a: build(*a), enumerate(variants)))

        labelled = truth is not None
        rows.sort(key=lambda r: r["score"], reverse=True)
        basis = ("correctness/completeness vs ground truth" if labelled
                 else "expected precision/recall under the model's confidence")
        result = {"labelled": labelled, "objective": objective,
                  "ranked_by": f"score = F{beta:g} of {basis}",
                  "best": rows[0]["id"], "candidates": rows}
        (root / "index.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
        return result

    def apply_candidate(self, candidate_id: str) -> dict:
        """Make a scored candidate the live result (re-stitch + re-vectorise)."""
        idx_path = self.dir / "candidates" / "index.json"
        if not idx_path.exists():
            raise RuntimeError("no candidates yet - run try_candidates first")
        index = json.loads(idx_path.read_text(encoding="utf-8"))
        row = next((r for r in index["candidates"] if r["id"] == candidate_id), None)
        if row is None:
            raise ValueError(f"no candidate {candidate_id!r}; have "
                             f"{[r['id'] for r in index['candidates']]}")
        params = dict(row["params"])
        threshold = params.pop("threshold")
        out = {"applied": candidate_id, "params": row["params"],
               "stitch": self.stitch(threshold=threshold),
               "vector": self.vectorize(**params)}
        if self.has_truth():
            out["metrics"] = self.evaluate()
        return out

    def score_network(self, tolerance_m: float = 5.0) -> dict:
        from gisagent.evaluate.network import score_network

        if not self.has_truth():
            raise RuntimeError("no ground truth available for this job")
        return score_network(self.network(), self.truth_path, tolerance_m=tolerance_m)

    # -- stage 6: evaluate -------------------------------------------------- #

    def evaluate(self, slack_px: int = 3) -> dict:
        from gisagent.evaluate.metrics import compare_masks

        if not self.has_truth():
            raise RuntimeError("no ground truth available for this job")
        t0 = time.perf_counter()
        with rasterio.open(self.mask_path) as ds:
            pred = ds.read(1) > 127
        with rasterio.open(self.truth_path) as ds:
            truth = ds.read(1) > 127
        m = compare_masks(pred, truth, slack_px=slack_px)
        result = m.to_dict()
        self.manifest["metrics"] = result
        self.record("evaluated", {"slack_px": slack_px}, result,
                    time.perf_counter() - t0)
        return result

    def sweep_threshold(self, thresholds: list[float] | None = None,
                        slack_px: int = 3) -> list[dict]:
        """Score the stitched confidence map at several cut-offs."""
        from gisagent.evaluate.metrics import sweep_threshold as _sweep

        if not self.has_truth():
            raise RuntimeError("no ground truth available for this job")
        with rasterio.open(self.conf_path) as ds:
            conf = ds.read(1).astype(np.float32)
        with rasterio.open(self.truth_path) as ds:
            truth = ds.read(1) > 127
        rows = _sweep(conf, truth, thresholds=thresholds, slack_px=slack_px)
        self.manifest["threshold_sweep"] = rows
        self.save()
        return rows


# --------------------------------------------------------------------------- #
# job management
# --------------------------------------------------------------------------- #

def jobs_root() -> Path:
    root = get_settings().outputs_dir / "jobs"
    root.mkdir(parents=True, exist_ok=True)
    return root


def new_job(name: str | None = None) -> Job:
    job_id = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    job = Job(jobs_root() / job_id)
    job.manifest = {
        "job_id": job_id,
        "name": name or job_id,
        "created_at": _utc(),
        "stage": "created",
        "stages": [],
    }
    job.save()
    return job


def get_job(job_id: str) -> Job:
    path = jobs_root() / job_id
    if not path.exists():
        raise FileNotFoundError(f"no such job: {job_id}")
    return Job(path)


def list_jobs() -> list[dict]:
    out = []
    for d in sorted(jobs_root().iterdir(), reverse=True):
        if (d / "job.json").exists():
            try:
                out.append(Job(d).status())
            except Exception:
                continue
    return out
