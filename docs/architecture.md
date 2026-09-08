# Architecture

## The problem

Annotating roads on satellite imagery is normally hand work: an operator loads a
raster into QGIS, traces road centrelines, and repeats. This system automates the
tracing and, more importantly, automates the *judgement* around it — choosing
prompts, thresholds and which areas need a second pass.

## Flow

```
Mass Roads index (train/valid/test)
        |  filenames encode a 100 m grid -> contiguous blocks chosen without downloading
        v
  region mosaic (GeoTIFF, EPSG:26986, 1 m/px)
        |  overlapped chipping
        v
  chips (1024 px, 128 px overlap)
        |  SAM 3 promptable concept segmentation ("road network")
        v
  per-chip confidence maps (float32)
        |  feathered stitch -> single confidence raster
        v
  confidence.tif  --threshold-->  mask.tif
        |  morphology: despeckle, fill holes, skeletonise
        v
  roads.geojson (centrelines, EPSG:26986 + WGS84)
        |
        +--> evaluated against ground-truth masks (IoU / precision / recall / F1)
```

Every stage writes into a job directory and appends a `StageRecord`, so a job is
resumable and fully auditable after the fact.

## Why these choices

**`qgis_process`, not a QGIS plugin.** The popular `qgis_mcp` servers on GitHub
(jjsantos01, nkarasiak, anitagraser) all work by running a socket server inside
QGIS Desktop, started by a human clicking a menu item. That cannot be
containerised or driven from a web backend. `qgis_process` is the supported
headless CLI, ships with every QGIS install, and is exposed by the official
`qgis/qgis` Docker image.

**SAM 3, not SAM 1/2.** SAM 3 does Promptable Concept Segmentation: a text phrase
returns instance masks for matching objects. SAM 1 and 2 only take points and
boxes, so road extraction with them means generating class-agnostic blobs and
then guessing which are roads. The phrase is a tunable parameter because concept
models are wording-sensitive.

**Upscaling before inference.** At 1 m/px a residential road is roughly 8 px
wide — far thinner than the objects SAM was trained on. Upsampling a chip before
inference puts roads at a more familiar scale, and measurably recovers roads the
model otherwise misses. It costs GPU memory, which is why it is per-call.

**Confidence maps, not hard masks.** Chips overlap, so the stitcher needs to
blend. Keeping float confidence until after stitching means the threshold stays a
downstream decision the agent can sweep without re-running inference — by far
the most expensive step.

**Contiguity from filenames.** `10378780_15` decodes to grid key `(1037, 8780)`,
and `origin = key * 100 + offset`. Tiles span exactly 1500 m and neighbours
differ by 15 grid units, so adjacent tiles abut with no gap or overlap. This was
verified against four published tile headers, and it means a contiguous region
can be assembled without downloading anything to find out.

## The agent

An OpenAI-compatible tool-calling loop (`agent/loop.py`) over the MCP tool
surface. It is not a fixed script: the model decides which tools to call and
when to iterate. The loop that matters is

```
segment -> stitch -> evaluate -> (sweep_threshold | refine_area) -> re-evaluate
```

`evaluate_result` and `low_confidence_roads` give the agent measured feedback, so
refinement is driven by numbers rather than guesswork. `refine_area` re-runs only
the chips intersecting a window, at a different prompt/threshold/upscale — the
automated equivalent of an operator zooming into a bad patch and re-tracing it.

Tool results are summarised before entering the transcript (`summarise`), because
raw geospatial payloads would otherwise exhaust the context window.

## MCP tool surface

16 tools in `mcp_servers/roads_server.py`:

| group | tools |
|---|---|
| discovery | `list_available_tiles`, `list_jobs`, `get_job_status` |
| region | `create_region_job`, `tile_region`, `inspect_chip` |
| inference | `segment_chips`, `stitch_result`, `refine_area` |
| quality | `evaluate_result`, `sweep_threshold`, `low_confidence_roads` |
| vector | `vectorize_result` |
| QGIS escape hatch | `qgis_version`, `list_qgis_algorithms`, `run_qgis_algorithm` |

The last group matters: rather than wrapping every QGIS algorithm, the agent can
list and invoke any of the ~1500 available ones directly.

## Web application

Single-page workspace (`web/index.html`) in three columns:

- **left** — region selection, layer toggles, accuracy metrics, vector stats
- **centre** — Leaflet map: imagery, predicted mask, ground truth, vector
  centrelines, with an area-picker for targeting refinement
- **right** — the agent conversation and a live trace of tool calls

State streams over `WS /ws/jobs/{job_id}`, so tool calls appear as they happen
rather than after the run finishes.

Leaflet is vendored under `web/static/vendor/`, so the UI has no CDN dependency
and no build step — which is also why there is no Node toolchain in the image.

## Known limits

- Zero-shot segmentation is the point of the design, but it is not a trained road
  extractor. Strict IoU around 0.46 reflects centreline placement being a few
  pixels off more than roads being missed — recall is 0.82 and relaxed F1 is 0.82.
- Precision is the weak axis (0.51 strict): parking lots, driveways and wide
  paths read as road to a concept model.
- Ground-truth labels only exist for the Mass Roads tiles. Uploaded imagery runs
  the same pipeline but cannot be scored.
