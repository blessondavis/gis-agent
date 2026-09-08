# gis-agent

Agentic road annotation on satellite imagery.

Tracing roads off aerial imagery is normally manual: load a raster in QGIS, draw
centrelines, repeat. This automates both the tracing *and* the judgement around
it — an LLM agent drives a SAM 3 segmentation pipeline over MCP tools, measures
its own output against ground truth, and re-runs the areas it got wrong.

![stack](https://img.shields.io/badge/QGIS-headless-green) ![stack](https://img.shields.io/badge/SAM-3-blue) ![stack](https://img.shields.io/badge/MCP-16%20tools-orange)

## What it does

```
Mass Roads GeoTIFFs -> region mosaic -> overlapped chips -> SAM 3 ("road network")
   -> feathered stitch -> threshold -> skeletonise -> road centrelines (GeoJSON)
   -> scored against ground truth -> agent refines the weak areas
```

The agent is not running a fixed script. It picks prompts, thresholds and
upscale factors, reads back measured IoU/F1, and calls `refine_area` on regions
that scored badly — the automated equivalent of an operator zooming into a bad
patch and re-tracing it.

## Results

Zero-shot on a 3 x 3 km region near Andover, Massachusetts (4 tiles, 1 m/px):

| metric | strict | relaxed (3 px) |
| --- | --- | --- |
| precision | 0.510 | 0.759 |
| recall | 0.825 | 0.893 |
| F1 | 0.631 | **0.821** |
| IoU | 0.460 | — |

85 road features, 15.8 km of centrelines. The gap between strict and relaxed
scores is the honest story here: the roads are *found* (recall 0.83) but
centreline placement is a few pixels off, and parking lots and driveways read as
road to a concept model. This is a zero-shot foundation model, not a trained road
extractor.

## Quickstart

Requires Python 3.12+, [uv](https://docs.astral.sh/uv/), and QGIS (for the
`qgis_process` CLI — the installer's default location is auto-detected).

```bash
uv sync --extra dev
cp .env.example .env      # then fill in the two keys
uv run gisagent doctor    # verifies GPU, QGIS, SAM 3 access and tool calling
```

`doctor` should show all-green before you go further. It checks that the LLM
actually *emits a tool call*, not merely that the key authenticates — a model
that cannot call tools will fail silently otherwise.

```bash
uv run gisagent regions --split train --size 2   # find a good block of tiles
uv run gisagent build 22529485_15 22679485_15 22529470_15 22679470_15 --name andover
uv run gisagent run <job-id>                     # tile -> segment -> vectorize -> score
uv run gisagent serve                            # web UI at http://127.0.0.1:8000
```

### Configuration

Both credentials go in `.env` (git-ignored):

| variable | purpose |
| --- | --- |
| `OPENAI_API_KEY` / `OPENAI_BASE_URL` | any OpenAI-compatible endpoint — OpenAI, NVIDIA NIM, vLLM, Groq, OpenRouter |
| `GISAGENT_LLM_MODEL` | must support tool calling |
| `HF_TOKEN` | `facebook/sam3` is gated; accept the terms on the model page first |
| `GISAGENT_DEVICE` | `auto` \| `cuda` \| `cpu` |

RTX 50-series (Blackwell, `sm_120`) needs CUDA 12.8+ wheels; `pyproject.toml`
already pins the `cu128` index for `torch`, since the default PyPI build has no
`sm_120` kernels and fails at runtime rather than at install.

## Web application

Single page, three columns:

- **left** — region selection, layer toggles, live accuracy, vector stats
- **centre** — Leaflet map: imagery, predicted mask, ground truth and vector
  centrelines, plus a drag-to-select box for targeting refinement
- **right** — agent conversation with a live trace of every tool call

Tool calls stream over a WebSocket, so you watch the agent work rather than
waiting for a final result. Leaflet is vendored — no CDN, no build step, no Node.

## Docker

```bash
docker compose -f docker/docker-compose.yml up --build
```

Built on the official `qgis/qgis` image, which ships the headless `qgis_process`
CLI. Defaults to CPU torch; for GPU set `TORCH_VARIANT: cu128` and uncomment the
`deploy.resources` block (needs `nvidia-container-toolkit`).

> Not yet run end-to-end — Docker is not installed on the development machine.
> The image definition is complete but unverified.

## Layout

```
src/gisagent/
  dataset/mass_roads.py   tile index, grid decoding, screening, download
  raster/                 georeferencing, chipping, stitching, previews
  segment/sam3.py         SAM 3 promptable concept segmentation
  vector/roads.py         morphology -> skeleton -> centreline GeoJSON
  evaluate/metrics.py     IoU / precision / recall / F1, strict and relaxed
  qgis/process.py         headless qgis_process wrapper
  mcp_servers/            16 MCP tools over the pipeline
  agent/loop.py           OpenAI-compatible tool-calling loop
  api/app.py              FastAPI + WebSocket
  pipeline.py             Job: stages, artefacts, manifest
```

See [docs/architecture.md](docs/architecture.md) for why each piece is built the
way it is — particularly why this drives `qgis_process` rather than the
plugin-over-a-socket approach the popular QGIS MCP servers use.

## Data

[Massachusetts Roads Dataset](https://www.cs.toronto.edu/~vmnih/data/) (Mnih).
Tiles are GeoTIFFs in EPSG:26986 at 1 m/px, 1500 x 1500 px, with ground-truth
road masks. Filenames encode position on a 100 m grid, so contiguous regions are
selected without downloading anything first.
