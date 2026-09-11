<p align="center">
  <img src="docs/assets/hero.jpg" alt="gis-agent: road centrelines traced by the agent over Boston Back Bay" width="100%">
</p>

<p align="center">
  <a href="#-quickstart"><img src="https://img.shields.io/badge/python-3.12%20%7C%203.13-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.12 | 3.13"></a>
  <a href="https://modelcontextprotocol.io"><img src="https://img.shields.io/badge/MCP-23%20tools-7c5cff?style=for-the-badge" alt="MCP: 23 tools"></a>
  <a href="https://huggingface.co/facebook/sam3"><img src="https://img.shields.io/badge/SAM%203-text--prompted-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black" alt="SAM 3"></a>
  <a href="https://qgis.org"><img src="https://img.shields.io/badge/QGIS-headless-589632?style=for-the-badge&logo=qgis&logoColor=white" alt="QGIS headless"></a>
  <a href="https://pytorch.org"><img src="https://img.shields.io/badge/PyTorch-cu128-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch"></a>
  <a href="LICENSE.md"><img src="https://img.shields.io/badge/license-PolyForm%20Noncommercial-4fe3e8?style=for-the-badge" alt="License: PolyForm Noncommercial 1.0.0"></a>
  <a href="https://github.com/blessondavis/gis-agent/stargazers"><img src="https://img.shields.io/github/stars/blessondavis/gis-agent?style=for-the-badge&logo=github&color=ffb84d" alt="GitHub stars"></a>
</p>

<p align="center">
  <b><a href="#-quickstart">Quickstart</a></b> ·
  <b><a href="#-finish-it-by-hand">Finish it by hand</a></b> ·
  <b><a href="#-the-harness">Harness</a></b> ·
  <b><a href="#-results">Results</a></b> ·
  <b><a href="#-beyond-massachusetts">Global test</a></b> ·
  <b><a href="#-mcp-tools">MCP tools</a></b> ·
  <b><a href="docs/harness.md">Design notes</a></b>
</p>

---

Tracing roads off aerial imagery is normally hand work: load a raster into QGIS,
draw centrelines one by one, repeat. **gis-agent automates the tracing *and* the
judgement around it.** An LLM agent drives a segmentation pipeline through MCP
tools, scores its own output, and re-runs the areas it got wrong.

You give it a region. It gives you back **road centrelines as GeoJSON**, an
**accuracy score**, and a **live trace** of every decision it made on the way.

<table>
  <tr>
    <td width="50%" valign="top">
      <h3>🤖 An agent with a harness</h3>
      It plans where you can see it, compares settings side by side before
      touching anything, and can't sign off until it has measured the result.
      Every turn can be rewound, files included.
    </td>
    <td width="50%" valign="top">
      <h3>✍️ You finish the last stretch</h3>
      Draw, reshape and delete roads in the same map. Your edits sit on top of
      the model's roads and survive every re-run. A review queue walks you
      through the roads it probably missed.
    </td>
  </tr>
  <tr>
    <td valign="top">
      <h3>🧭 Judges itself without labels</h3>
      On imagery nobody has labelled, it scores its own output with a
      vision-model critic, network topology and its own confidence, each
      checked against ground truth before being trusted.
    </td>
    <td valign="top">
      <h3>🗺️ Real GIS output</h3>
      A noded road network as WGS84 GeoJSON, with every junction a real
      node. Headless QGIS, with all 712 algorithms callable directly.
    </td>
  </tr>
</table>

## 🛰️ See it work

<p align="center">
  <img src="docs/assets/pipeline-strip.jpg" alt="Four stages on Boston Back Bay: imagery, road confidence, centrelines, and agreement with ground truth" width="100%">
</p>

One real 3 × 3 km run over Boston Back Bay: **2,125 centrelines** extracted and
scored against the dataset's ground truth at **relaxed F1 0.79**. Green is
correct, orange is a false positive, red is a missed road.

---

## ✍️ Finish it by hand

The model gets most of the network. The last stretch is faster to draw than to
prompt for, so the workspace has real editing tools, right in the same map:

<table>
  <tr>
    <td width="50%"><img src="docs/assets/editor-review.jpg" alt="Review mode stepping through a suggested gap connector at an intersection"></td>
    <td width="50%"><img src="docs/assets/editor-draw.jpg" alt="Drawing a road, with vertices snapping to the existing network"></td>
  </tr>
  <tr>
    <td valign="top"><b>Review suggestions</b> <kbd>R</kbd>: step through roads the network is
      probably missing, then <kbd>A</kbd> to accept or <kbd>X</kbd> to dismiss.</td>
    <td valign="top"><b>Draw</b> <kbd>D</kbd>, <b>reshape</b> <kbd>V</kbd>, <b>delete</b>
      <kbd>X</kbd>, with vertices snapping to junctions and roads. <kbd>Ctrl</kbd>+<kbd>Z</kbd> undoes.</td>
  </tr>
</table>

- **Your edits survive the agent.** They're kept as their own undoable layer
  and merged over the model's roads every time the network is read. The agent
  can re-run the model underneath you and nothing you drew is lost.
- **Junctions are real.** A road you end on another snaps to it and splits
  it there, so the delivered GeoJSON is a noded network rather than lines that
  merely touch.
- **Suggestions are ranked by how often they're right.** Measured on Boston,
  80% of the top-ranked *gap* suggestions (two dead ends that nearly meet) are
  real roads. "Possible road" suggestions are right about half the time and
  are labelled that way.
- **The Completion panel shows who did what.** It gives the share of the real
  road network found, by the model alone and with your edits.

---

## 🧠 The harness

The agent loop borrows its structure from xAI's open-source
[Grok Build](https://github.com/xai-org/grok-build), adapted to rasters and road
networks instead of code:

| | |
| --- | --- |
| **Autonomous mode** | **Annotate autonomously** (or `gisagent annotate <job>`) hands the agent the whole region under rules the harness enforces in code: pipeline order, budgets, no repeated calls, every change measured by the harness, the best result kept, a plateau stop, and a definition of done it can't finish without. It ends with a report whose verdict is computed, not claimed. Unattended on Boston **with the labels hidden**, it beat the default pipeline when checked against the truth afterwards: precision objective 0.793 → 0.807, better on both correctness and completeness. |
| **Plan first** | Tick *Plan first* and the agent can only look and measure. It proposes a plan, and nothing changes until you click **Approve & run**. |
| **Visible plan** | A checklist pinned above the chat, kept current as the agent works. |
| **Candidates, then apply** | `try_candidates` builds several settings side by side without touching the live result, and ranks them by what you asked for: precision, recall or balanced. |
| **Stop gate** | The agent can't finish while its output has changed since it last measured, or while its reply quotes a score no tool returned. |
| **Rewind, files included** | ↺ on any message restores the mask, centrelines, confidence maps, your edits and the plan to just before it. Grok Build's rewind leaves files alone. |
| **Session replay** | The whole trace (plan, tool cards, checks, edits) rebuilds after a reload. |
| **Headless** | `gisagent agent <job> "..." --json` runs the same harness from scripts and CI. |

Ranking candidates on unlabelled imagery needs a judge that works without
labels, so two were tested against ground truth first. **Topology ranked them
backwards (Spearman −0.64).** The network's expected precision and recall under
the model's own confidence picked the ground-truth winner under every objective
(ρ 0.89 to 1.00). The measurements, the bugs the live runs caught, and the
limits are written up in [docs/harness.md](docs/harness.md).

---

## 🔧 How it works

<p align="center">
  <img src="docs/assets/architecture.svg" alt="The LLM agent calls 23 MCP tools that drive six stages: region, chip, segment, stitch, vectorise, judge, with a feedback loop into segmentation" width="100%">
</p>

The agent is **not** running a fixed script. It chooses prompts, thresholds and
upscale factors, reads back measured IoU/F1, and calls `refine_area` on regions
that scored badly.

<details>
<summary><b>The pipeline, stage by stage</b></summary>

```
Massachusetts Roads index
        │  filenames encode a 100 m grid, so contiguous regions are
        │  chosen without downloading anything first
        ▼
  region mosaic ......................  GeoTIFF, EPSG:26986, 1 m/px
        │  overlapped chipping
        ▼
  chips ..............................  1024 px, 128 px overlap
        │  SAM 3 ("road network") or the trained U-Net
        ▼
  per-chip confidence maps ...........  float32, not hard masks
        │  feathered stitch
        ▼
  confidence.tif ──threshold──▶ mask.tif
        │  despeckle → fill holes → skeletonise
        ▼
  roads.geojson ......................  centrelines
        │
        └──▶ judged (metrics / critic / topology) ──▶ agent refines the weak areas
```

</details>

The full reasoning behind each design decision is in
[docs/architecture.md](docs/architecture.md). The short version:

| choice | why |
| --- | --- |
| `qgis_process` CLI | The popular QGIS MCP servers need a **Desktop GUI** with a human clicking "Start Server". That can't be containerised or driven from a web backend. |
| SAM 3, not SAM 1/2 | SAM 3 takes a **text prompt**. SAM 1/2 only take points and boxes, so roads mean generating blobs and guessing which are roads. |
| Confidence maps, not masks | Chips overlap and must blend. Keeping float confidence means the threshold can be swept **without re-running inference**, which is by far the slowest step. |
| Swappable segmenters | `UNetRoadSegmenter` and `Sam3RoadSegmenter` share one interface, so the pipeline, MCP tools and web app don't care which one runs. |

---

## 🧭 How it knows when to stop

<p align="center">
  <img src="docs/assets/quality-signals.svg" alt="Four quality signals: ground truth, VLM critic, topology and model confidence" width="100%">
</p>

Ground truth only exists for the benchmark tiles. On real work there is none, so
something else has to tell the loop whether an annotation is good enough.

That "something" was **tested before it was built on.** Six overlays of known
quality were shown blind to a vision model, which ranked them at **Spearman
ρ = +0.886** against true IoU. It also exposed a blind spot: a uniformly shifted
annotation still *looks* right. Topology catches exactly that case, so the two
ship together. The full study is in [docs/vlm-critic.md](docs/vlm-critic.md).

---

## 📊 Results

<p align="center">
  <img src="docs/assets/benchmark.svg" alt="Bar charts comparing SAM 3 and U-Net on IoU, F1 and recall for suburban Andover and urban Boston" width="100%">
</p>

Measured against the dataset's own ground-truth masks. **Relaxed** allows 3 px of
slack, roughly the disagreement between two human annotators tracing the same
road.

| region | road % | model | IoU | F1 | precision | recall | relaxed F1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Andover (suburban) | 1.1 | sam3 | 0.455 | 0.626 | 0.497 | 0.844 | 0.822 |
| Andover (suburban) | 1.1 | **unet** | **0.581** | **0.735** | 0.642 | 0.860 | 0.857 |
| Boston (urban) | 15.9 | sam3 | 0.098 | 0.178 | 0.511 | 0.108 | 0.217 |
| Boston (urban) | 15.9 | **unet** | **0.453** | **0.624** | 0.621 | **0.627** | 0.791 |

In the urban case, **IoU improves 4.6× and recall 5.8×**. In vector terms the
same region goes from 149 centrelines totalling 13.3 km to 2,061 totalling
162.9 km. SAM 3 was finding the arterials and almost none of the grid.

<details>
<summary><b>🔍 Side by side on Boston Back Bay (full-resolution comparison)</b></summary>
<br>

![SAM 3 vs U-Net on Boston Back Bay](docs/compare_urban.png)

Top: zero-shot SAM 3. Bottom: the trained U-Net. Left is the extracted network,
right is agreement with ground truth.

</details>

### Where it falls down

- **Zero-shot SAM 3 degrades badly on dense street grids.** At 1 m/px, residential
  streets are ~10–15 px wide, shadowed by buildings and lined with parked cars,
  and a concept model reads them as texture between rooftops. What it *does* find
  is real (precision 0.51); it just misses about 90 % of it. The U-Net exists to
  fix this.
- **Parking lots, driveways and wide paths read as road.** Precision is the weak
  axis everywhere, though supervision helps (0.497 → 0.642 in the suburbs).
- **Strict IoU understates quality.** Centreline placement is a few pixels off
  more often than roads are actually missed, which is why relaxed metrics are
  reported alongside.
- **Only the Massachusetts tiles can be scored.** Uploaded imagery runs the same
  pipeline and gets the label-free signals, but no IoU.
- **135 tiles and ten minutes is a small budget.** Published work on this dataset
  reaches higher. The point is that the failure mode is fixable with supervision,
  and that swapping backends changes nothing else in the system.

---

## 🌍 Beyond Massachusetts

A model trained on one US state was then tested on the rest of the world, using
[Global-Scale](https://arxiv.org/abs/2411.16733) (CVPR 2025): 1 m/px imagery from
six continents, plus a test set of cities absent from its training pool.

<p align="center">
  <img src="docs/assets/global-results.svg" alt="Found, correct and centreline F1 before and after fine-tuning, on Massachusetts, global test sites and unseen cities" width="100%">
</p>

**Out of the box, it fails abroad.** On unseen cities it finds 21% of the road
network, against 82% at home. It isn't a resolution problem: shrinking or
enlarging the imagery before inference only made things worse. What it does
draw is mostly right (74% correct), so it's under-detection. The model has
learned what a Massachusetts road looks like.

**Ten minutes of fine-tuning closes much of the gap.** The fine-tune starts
from the Massachusetts model and mixes Massachusetts and Global-Scale crops
50/50. The best epoch is chosen on the mean of the two validation sets, so the
new domain can't be learned at the old one's expense:

| held-out set | found | correct | centreline F1 |
| --- | --- | --- | --- |
| Massachusetts (20 tiles) | 0.817 → 0.816 | 0.885 → 0.887 | 0.850 → 0.850 |
| Global test sites (60) | 0.125 → **0.567** | 0.841 → 0.744 | 0.218 → **0.644** |
| Unseen cities (130) | 0.210 → **0.539** | 0.737 → 0.567 | 0.327 → **0.553** |

<p align="center">
  <img src="docs/assets/global-ood.jpg" alt="Lucerne and Shenzhen tiles: imagery, zero-shot prediction and fine-tuned prediction coloured by agreement with ground truth" width="100%">
</p>

The tiles above are the ones closest to each city's *median* improvement, not
the best. They show the trade honestly. Lucerne goes from 33% found to 69%.
Shenzhen goes from 55% to 81% found, but at 33% correct. Many of those orange
"false positives" follow narrow alleys between buildings that OpenStreetMap,
and therefore the labels, doesn't include. That's plausible, but it can't be
scored here, so treat the correctness drop as real.

<details>
<summary><b>How the test was kept honest</b></summary>
<br>

- **The official split leaks.** 71 of Global-Scale's official training tiles are
  byte-identical to validation or test tiles, and many more overlap them
  spatially at a partial offset. Tiles were grouped into *sites* by shared road
  junctions, and no fine-tuning site overlaps any test site.
- **Labels were redrawn to match.** Global-Scale ships road graphs whose own
  masks are 3 px wide. Massachusetts labels measure 7 px, so the graphs were
  re-rasterised at 7 px. The axis order was verified at IoU 0.999 against the
  dataset's own rendering.
- **The headline metrics ignore label width.** Found and correct compare
  thinned centrelines by length, within 5 px, so a label convention can't
  inflate or deflate them. Pixel IoU is in the JSON reports too.

Reproduce it: prepare the data with `gisagent.dataset.global_scale.prepare()`
(about 3 GB), then

```bash
uv run gisagent train --skip-fetch --init models/unet_roads.pt \
    --extra-data <global-scale dir> --lr 1e-4 --epochs 12 --out-name unet_roads_global.pt
uv run gisagent evaluate-tiles --checkpoint models/unet_roads.pt \
    --checkpoint models/unet_roads_global.pt --mass-val \
    --data <global-scale dir> --set ood --set id_test
```

To use the fine-tuned model everywhere, set
`GISAGENT_UNET_CHECKPOINT=models/unet_roads_global.pt` in `.env`. Global-Scale
uses Google imagery and is published for research. It's fetched from an
unofficial Hugging Face mirror and is not redistributed here.

</details>

---

## 🚀 Quickstart

**Requirements:** Python 3.12 or 3.13 · [uv](https://docs.astral.sh/uv/) ·
QGIS 3.x (only the bundled `qgis_process` CLI is used, and it's auto-detected) ·
a GPU is optional but strongly recommended (~4 GB VRAM is enough).

```bash
git clone https://github.com/blessondavis/gis-agent.git
cd gis-agent
uv sync --extra dev
cp .env.example .env        # then fill in the two credentials below
```

| variable | how to get it |
| --- | --- |
| `HF_TOKEN` | `facebook/sam3` is **gated**. Accept the terms at [huggingface.co/facebook/sam3](https://huggingface.co/facebook/sam3), then make a read token at [settings/tokens](https://huggingface.co/settings/tokens) |
| `OPENAI_API_KEY` + `OPENAI_BASE_URL` | any OpenAI-compatible endpoint: NVIDIA NIM, OpenAI, vLLM, Groq and OpenRouter all work |

The model must support **tool calling**, or the agent can't do anything. Verified
working on NVIDIA NIM: `nvidia/nemotron-3-super-120b-a12b` and `openai/gpt-oss-20b`.

**Check everything before going further:**

```console
$ uv run gisagent doctor
component   status  detail
torch       ok      2.11.0+cu128 cuda=12.8 RTX 5050 sm_120 8.5 GB
qgis        ok      QGIS 3.44.14-Solothurn
sam model   ok      facebook/sam3 (token set)
llm         ok      nvidia/nemotron-3-super-120b-a12b
```

`doctor` sends a real tool-call probe, not just an auth check. Without it, a model
that authenticates but can't call tools would fail silently much later.

**Then three commands, start to finish:**

```bash
# 1. find a region worth annotating (ranked by road density, cached after first run)
uv run gisagent regions --split train --size 2

# 2. download those tiles and mosaic them into a job
uv run gisagent build 22529485_15 22679485_15 22529470_15 22679470_15 --name andover

# 3. segment → stitch → vectorise → score
uv run gisagent run <job-id>
```

`run` prints a metrics table and the path to `roads.geojson`.

> [!NOTE]
> The first SAM 3 run downloads **~3.4 GB of weights**. Later runs reuse the
> Hugging Face cache.

---

## 🖥️ The web app

```bash
uv run gisagent serve        # → http://127.0.0.1:8000
```

One page, three columns:

| column | what it does |
| --- | --- |
| **left** | pick a region, toggle layers, watch accuracy, vector stats and the quality verdict update |
| **centre** | Leaflet map with imagery, predicted mask, ground truth and centrelines. Drag a box to target a rework |
| **right** | talk to the agent and watch each tool call stream in as it happens |

Ask it things like *"annotate the roads in this region"*, *"try to improve the
score"*, or *"which roads are least confident?"* Select an area on the map, tell
it what's wrong there, and it re-runs just those chips.

Leaflet is vendored locally, so there's no CDN, no Node and no build step.

---

## 🔌 MCP tools

`gisagent mcp` speaks MCP over stdio, so Claude Desktop or any other MCP client
can drive the pipeline directly.

| group | tools |
| --- | --- |
| 🔎 discovery | `list_available_tiles`, `list_jobs`, `get_job_status` |
| 🧩 region | `create_region_job`, `tile_region`, `inspect_chip` |
| 🧠 inference | `segment_chips` (U-Net or SAM 3), `stitch_result`, `refine_area` |
| ⚖️ candidates | `try_candidates`, `apply_candidate` |
| 📏 quality, with labels | `evaluate_result`, `sweep_threshold`, `low_confidence_roads` |
| 🧭 quality, label-free | `critique_annotation`, `check_topology` |
| ✍️ the person's work | `network_status`, `suggest_missing_roads` |
| 🗺️ vector | `vectorize_result`, `repair_geometry` |
| 🛠️ QGIS | `qgis_version`, `list_qgis_algorithms`, `run_qgis_algorithm` |

The QGIS group is an escape hatch. Rather than wrapping every algorithm, the agent
can list and call any of the 712 QGIS algorithms directly. `repair_geometry`
reports topology **before and after** each fix (snap, remove dangles, extend,
simplify, smooth), so the agent can check that a repair actually helped instead
of assuming it did.

## 🧰 Commands

| command | purpose |
| --- | --- |
| `gisagent doctor` | verify GPU, QGIS, SAM 3 access and LLM tool calling |
| `gisagent regions` | rank contiguous tile blocks by road density |
| `gisagent build <tiles...>` | download tiles and mosaic them into a job |
| `gisagent run <job-id>` | full pipeline with metrics; `--model sam3\|unet` |
| `gisagent annotate <job-id>` | annotate the region autonomously under the task rules; `--objective precision\|recall\|balanced`, targets, budgets; writes `report.md` |
| `gisagent agent <job-id> "..."` | one agent turn, headless; `--plan` for read-only, `--json` for an event stream |
| `gisagent train` | train the U-Net; `--init` plus `--extra-data` fine-tunes on a second dataset |
| `gisagent evaluate-tiles` | score checkpoints on held-out tiles: pixel, relaxed and centreline metrics |
| `gisagent benchmark <jobs...>` | score backends against each other |
| `gisagent jobs` | list jobs, newest first |
| `gisagent serve` | run the web app |
| `gisagent mcp` | expose the 23 MCP tools on stdio, for an external client |

Add `--help` to any of them.

---

## 🧠 The supervised backend

Two backends ship together: zero-shot **SAM 3**, and a **U-Net trained on the
dataset's own labels**. The trained one is better everywhere and much better in
dense cities, and it is the default (`GISAGENT_BACKEND=unet`).

```bash
uv run gisagent train --tiles 160 --epochs 14      # ~1.4 GB download, ~10 min on an RTX 5050
uv run gisagent run <job-id> --model unet
uv run gisagent benchmark <suburban-job> <urban-job> --models sam3,unet
```

<details>
<summary><b>Why a U-Net, and not a SAM 3 fine-tune</b></summary>
<br>

The urban failure is not a tuning problem. SAM 3 is a *concept* model: it was
never shown what a road looks like from 400 m up, and a narrow shadowed street
between rooftops does not read as one. The Massachusetts Roads dataset contains
exactly that supervision (1,171 tiles of labelled roads at 1 m/px), so the fix is
to use it.

This trains a **ResNet-34 U-Net (24 M params) from scratch** with an ImageNet
encoder, not a fine-tune of SAM 3. There are two reasons:

1. SAM 3 is 0.9 B parameters. Fine-tuning it needs far more than 8 GB of VRAM.
2. It wouldn't help as much. The failure is that the model doesn't know what an
   aerial road is, and a small model learning that from scratch on 160 tiles is a
   better use of the same GPU-hour.

</details>

<details>
<summary><b>Training details that matter more than they look</b></summary>
<br>

`train` writes `models/unet_roads.pt` plus a `train_report.json`. Tiles are chosen
from the screening cache to span the useful road-density range. Near-empty tiles
cost disk without teaching anything, and downtown blocks where the labels paint
wide swathes skew the model towards predicting road everywhere.

- **Crops are biased towards windows containing road.** Roads are a few percent
  of pixels, so uniformly random crops are mostly empty and the model learns to
  predict background everywhere.
- **Blank-heavy crops are rejected.** Many tiles are edge-of-coverage and carry
  white no-data padding. Training on it teaches that white means background.

The measured run trained for 9.9 minutes on 135 tiles (val IoU 0.608 at epoch 13).
Reuse an existing download with `--skip-fetch`.

`UNetRoadSegmenter` has two honest differences from SAM 3, both inherent to a
supervised model:

- `prompt` is accepted and **ignored**, because there is no text conditioning.
- `n_instances` is 1 when anything is found. This is semantic, not instance,
  segmentation.

</details>

---

## 🐳 Docker

```bash
docker compose -f docker/docker-compose.yml up --build
```

Built on the official `qgis/qgis` image for the headless `qgis_process` CLI.
Defaults to CPU torch. For GPU, set `TORCH_VARIANT: cu128` in the compose file
and uncomment the `deploy.resources` block (needs `nvidia-container-toolkit`).

> [!WARNING]
> **Not yet verified end to end.** Docker isn't installed on the development
> machine, so the image definition is complete but untested.

---

## 📁 Project layout

```
src/gisagent/
  dataset/                Massachusetts Roads index + download; Global-Scale preparation
  raster/                 georeferencing, chipping, stitching, previews
  segment/                SAM 3 concept segmentation, U-Net sliding-window inference
  train/                  training-set selection, road-biased crops, training / fine-tuning
  vector/roads.py         morphology → skeleton → centreline GeoJSON
  vector/edits.py         the person's edit layer: merge, junction splitting, suggestions
  vector/topology.py      label-free network scoring: components, dangles, gaps
  critic/vlm.py           vision-model critic for unlabelled imagery
  evaluate/               pixel metrics, length-based network scores, cross-dataset tiles
  qgis/process.py         headless qgis_process wrapper
  mcp_servers/            23 MCP tools over the pipeline
  agent/loop.py           the harness: plan, plan mode, stop gate, working memory, spilling
  checkpoints.py          per-turn snapshots, so a rewind restores files
  api/app.py              FastAPI + WebSocket, edits, rewind, session replay
  pipeline.py             Job: stages, artefacts, manifest, candidates
web/                      single-page workspace + map editor, no build step
docker/                   Dockerfile + compose
docs/                     harness notes, architecture, VLM critic study, design brief
```

Run the tests with `uv run pytest` (119 tests, no network or GPU required;
the harness and its task rules are tested against a scripted fake model).

---

## 🛠️ Troubleshooting

<details>
<summary><code>doctor</code> says the LLM "responds, but did NOT emit a tool call"</summary>
<br>
The model can't call tools, so pick another. On NVIDIA NIM, many catalogue
entries return 404 unless they're enabled for your account.
</details>

<details>
<summary><code>CUDA error: no kernel image is available</code></summary>
<br>
RTX 50-series (Blackwell, <code>sm_120</code>) needs CUDA 12.8+ wheels.
<code>pyproject.toml</code> already pins the <code>cu128</code> index. If you
installed torch another way, reinstall from
<code>https://download.pytorch.org/whl/cu128</code>.
</details>

<details>
<summary><code>qgis_process not found on PATH</code></summary>
<br>
Set <code>GISAGENT_QGIS_PROCESS</code> in <code>.env</code> to the full path. On
Windows it's usually
<code>C:\Program Files\QGIS &lt;version&gt;\bin\qgis_process-qgis-ltr.bat</code>.
</details>

<details>
<summary><code>failed to hardlink file ... os error 396</code></summary>
<br>
uv's cache can't hardlink across OneDrive. Use
<code>uv sync --link-mode=copy</code>.
</details>

<details>
<summary>The region looks mostly blank</summary>
<br>
Some tiles sit at the edge of the survey area and are largely white no-data.
<code>gisagent regions</code> ranks alternatives, and <code>build</code> warns
when the tiles it fetched are mostly padding.
</details>

---

## 📚 Data

[Massachusetts Roads Dataset](https://www.cs.toronto.edu/~vmnih/data/) (Mnih,
2013): 1,171 aerial tiles, GeoTIFF in EPSG:26986 at 1 m/px, 1500 × 1500 px, with
ground-truth road masks. Released for research use.

## 📜 License

[PolyForm Noncommercial 1.0.0](LICENSE.md). You're free to use, modify and share
this for research, study, personal projects, and by non-profits, schools and
public bodies. **Commercial use needs a separate license**, so
[open an issue](https://github.com/blessondavis/gis-agent/issues) to ask.

The license covers this repository's code only. The SAM 3 weights come under
[Meta's SAM license](https://huggingface.co/facebook/sam3), and the Massachusetts
Roads data is released for research use.

## ⭐ Star history

<a href="https://star-history.com/#blessondavis/gis-agent&Date">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=blessondavis/gis-agent&type=Date&theme=dark">
    <img src="https://api.star-history.com/svg?repos=blessondavis/gis-agent&type=Date" alt="Star history chart" width="100%">
  </picture>
</a>

<p align="center">
  <sub>If this saved you an afternoon of tracing roads by hand, a ⭐ helps other people find it.</sub>
</p>
