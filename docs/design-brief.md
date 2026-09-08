# gis-agent — UI/UX design brief

A brief for designing the interface. It describes what the product does, who
uses it, every screen state and data shape that exists, and where the current
implementation falls short. Written to be handed to a designer (or pasted into a
design tool) without needing to read the code.

---

## 1. The product in one paragraph

**gis-agent turns satellite imagery into road maps, automatically.**

Mapping roads from aerial photography is normally hand work: a GIS analyst opens
a satellite image in QGIS and traces every road centreline with a mouse, for
hours. gis-agent replaces that with an AI agent. You pick a region, the agent
segments the roads, measures its own accuracy against reference data, notices
where it did badly, and re-runs those areas. You watch it work and correct it in
plain language: *"the roads in the top-left are missing — rework that area."*

The output is a **GeoJSON file of road centrelines** plus an accuracy score.

---

## 2. Who uses it, and what they need to feel

| user | goal | the feeling to design for |
| --- | --- | --- |
| **GIS analyst** (primary) | Get usable road vectors without tracing them by hand | *"I can trust this, and I can see exactly where not to."* |
| **Remote-sensing researcher** | Compare model quality across regions | *"The numbers are right there and honest."* |
| **Reviewer / manager** | Sanity-check output before it ships | *"I can see what's wrong in five seconds."* |

The emotional core is **trust through visible reasoning**. This is an AI making
judgement calls on someone's professional deliverable. Every automated decision
must be inspectable, and every accuracy claim must be visible next to the thing
it describes. The interface should never feel like a magic box.

The single most important design idea: **the map is the truth, and the agent's
work is shown on it.** Not in a log, not in a modal — on the imagery.

---

## 3. What actually happens (the mental model)

```
1  PICK      choose a region of satellite imagery      (4 tiles, 3 × 3 km)
2  SLICE     cut it into overlapping chips             (16 chips, 1024 px)
3  SEGMENT   AI finds road pixels in each chip         ← slow, ~1–2 min
4  STITCH    blend chips back into one image
5  TRACE     convert road blobs to centrelines         → GeoJSON
6  SCORE     compare against reference data            → IoU / F1
7  REFINE    agent re-runs the areas that scored badly  ↺ back to 3
```

Steps 2–7 are driven by the agent, which decides what to do and reports back.
The user can also drive any step manually.

A **job** is one region moving through those stages. Its `stage` value is exactly
one of: `created → tiled → segmented → stitched → vectorized → evaluated`.
Design a progress representation for these six states. Note that `refine` loops
back, so this is **not** a strictly linear stepper — a job can return to
`segmented` after being `evaluated`.

---

## 4. Screen inventory

Currently one screen with three columns. **The designer should feel free to
restructure this** — see §10 for known problems with the current layout.

### 4.1 Left rail — region, layers, quality

**Region picker**
- List of jobs, newest first. Each shows: name, stage, tile count, IoU if scored.
- `New region` opens a dialog: split (`train`/`valid`/`test`), block size
  (2×2 / 3×3 / 4×4), a checkbox for "screen out blank / road-poor tiles", and an
  optional name.
- `Upload` accepts a user's own georeferenced `.tif` via drag-and-drop. Uploaded
  imagery has **no reference data**, so it can never be scored — the accuracy
  panel must degrade gracefully, not show zeros.

**Layer toggles** — five layers over the map, each independently switchable:

| layer | what it is | current colour |
| --- | --- | --- |
| Satellite imagery | the input photo | — |
| Predicted mask | AI's road pixels | red |
| Ground truth | reference road pixels | green |
| Road centrelines | the vector output | yellow |
| OSM basemap | context outside the region | — |

Plus an **overlay opacity** slider (0–100 %, currently defaults to 70 %).

> ⚠️ **Colour is doing critical work here and the current choice is bad.** Red =
> prediction and green = truth is unreadable for the ~8 % of men with red–green
> colour blindness, and it collides with the natural reading of red = error /
> green = correct. Needs rethinking. See §9.

**Accuracy panel** — six numbers, in two rows of three:

```
IoU 0.581    F1 0.735    relaxed F1 0.857
precision 0.642   recall 0.860   relaxed recall 0.893
```

These need explaining without a manual. Suggested plain-language framings:
- **recall** → "found 86 % of the roads"
- **precision** → "64 % of what it drew is really a road"
- **relaxed** → "allowing 3 px of slack, about the disagreement between two human
  tracers"
- **IoU** → the strict overlap score; the one researchers want, the one a
  layperson will misread as "58 % correct"

**Vector stats** — `n_features`, `total_length_km`, `mean_confidence`,
`low_confidence_features`, and a download button for the GeoJSON.

### 4.2 Centre — the map

Leaflet map showing the imagery with the enabled overlays. Also:
- `Select area to rework` — drag a box on the map, which becomes the target for
  the next refinement instruction.
- A legend for the active layers.

**This is the hero of the product and currently gets the least design attention.**

### 4.3 Right rail — the agent

- A chat thread with the agent.
- A **live trace**: every tool call streams in as it happens over a WebSocket.
- Status pill: `idle` / `thinking` / `working`.
- Suggestion chips, currently: *"Annotate the roads in this region"*, *"Try to
  improve the score"*, *"Which roads are least confident?"*, and when an area is
  selected: *"The roads in this area are missing — rework it"*, *"This area has
  too many false roads"*.
- `Reset` clears the conversation.

---

## 5. The agent trace — the most interesting design problem

The agent streams typed events over a WebSocket. **Nine event types**, all of
which need a visual treatment:

| event | meaning | frequency |
| --- | --- | --- |
| `connected` | socket opened | once |
| `status` | e.g. "16 tools available" | rare |
| `thinking` | model is deciding | between every tool call |
| `tool_call` | calling a tool, with arguments | many |
| `tool_result` | what the tool returned | many |
| `message` | prose from the agent | occasional |
| `done` | finished, with a summary | once |
| `error` | something failed | rare |
| `eof` | stream closed | once |

The agent calls **16 tools**, and these are the verbs the user will see scroll
past. They deserve iconography and grouping, not raw names:

| group | tools |
| --- | --- |
| Discovery | `list_available_tiles`, `list_jobs`, `get_job_status` |
| Region | `create_region_job`, `tile_region`, `inspect_chip` |
| Inference | `segment_chips`, `stitch_result`, `refine_area` |
| Quality | `evaluate_result`, `sweep_threshold`, `low_confidence_roads` |
| Vector | `vectorize_result` |
| QGIS | `qgis_version`, `list_qgis_algorithms`, `run_qgis_algorithm` |

**Design questions worth solving:**

- A full annotation run is **20–40 tool calls**. Raw, that's an unreadable wall.
  How do you collapse repetitive calls (16 × `segment_chips`) into one legible
  progress unit while keeping detail available?
- Tool results carry real numbers (IoU went 0.098 → 0.453). Those are the
  interesting moments in the whole stream. How do they stand out from routine
  calls?
- `refine_area` is the agent noticing its own mistake and fixing it. That is the
  product's best story and should feel like a beat, not another log line.
- Slow steps (`segment_chips` takes 1–2 minutes) need progress, not a spinner.
  Per-chip completion is available.

---

## 6. Real data to design against

Do not design against lorem ipsum — these are actual values from real runs.

**Two regions, deliberately different:**

| | Andover, MA | Boston Back Bay, MA |
| --- | --- | --- |
| character | suburban, wooded | dense city grid |
| road coverage | 1.1 % of pixels | 15.9 % |
| area | 3 × 3 km, 4 tiles | 3 × 3 km, 4 tiles |
| centrelines found | 85 | 2,061 |
| total length | 15.8 km | 162.9 km |

**Two AI backends, with genuinely different quality:**

| region | model | IoU | F1 | precision | recall |
| --- | --- | --- | --- | --- | --- |
| Andover | SAM 3 (zero-shot) | 0.455 | 0.626 | 0.497 | 0.844 |
| Andover | U-Net (trained) | 0.581 | 0.735 | 0.642 | 0.860 |
| Boston | SAM 3 (zero-shot) | 0.098 | 0.178 | 0.511 | 0.108 |
| Boston | U-Net (trained) | 0.453 | 0.624 | 0.621 | 0.627 |

**That third row is the most important row in this document.** IoU 0.098 means
the AI almost completely failed. The interface must communicate that *loudly and
unmistakably*, because a user who ships that output has shipped a map with 90 %
of the streets missing. A number in a small grey box is not enough.

**Design a "quality verdict" treatment** with roughly these bands:

| IoU | verdict | what the user should do |
| --- | --- | --- |
| > 0.55 | good | review and export |
| 0.35 – 0.55 | usable, needs review | check the false positives |
| 0.15 – 0.35 | poor | refine before using |
| < 0.15 | failed | do not ship this |

---

## 7. Empty, loading, and error states

Every one of these occurs in normal use:

- **No jobs yet.** First-run experience. The user has nothing and must be led to
  "create a region". This is the most important empty state.
- **Job created, nothing computed.** Imagery only, no overlays, no score.
- **Segmenting.** 1–2 minutes of real work. Per-chip progress available.
- **Scored but no reference data** (uploaded imagery). Accuracy panel must say
  "cannot be scored" — never "0.000".
- **Region is mostly blank.** Some tiles are edge-of-coverage and largely white
  no-data. The app detects this; the UI should warn rather than silently show a
  half-empty map.
- **Model not configured.** Missing `HF_TOKEN` or API key. `/api/health` reports
  this; the UI shows status badges.
- **Agent error.** Tool failed, or the LLM can't call tools.
- **Low-confidence features.** Some centrelines are flagged uncertain — these
  should be visually distinguishable on the map, not just counted.

Health badges currently in the header, from `GET /api/health`:
`{ok, llm_configured, llm_model, sam_model, sam_token, qgis, cuda, gpu, device}`

---

## 8. Data contracts

Everything the UI can display.

**Job object** (`GET /api/jobs`)
```json
{
  "job_id": "20260908-093543-2152ac",
  "stage": "vectorized",
  "region": {
    "width": 3000, "height": 3000, "bands": 3,
    "crs": "EPSG:26986", "n_tiles": 4,
    "truth": true, "truth_fraction": 0.158758,
    "bounds_wgs84": [-71.093674, 42.334079, -71.057084, 42.361221]
  },
  "artifacts": {
    "image": true, "truth": true, "chips": true,
    "confidence": true, "mask": true, "vectors": true
  },
  "metrics": {
    "iou": 0.453, "precision": 0.621, "recall": 0.627, "f1": 0.624,
    "relaxed_precision": 0.794, "relaxed_recall": 0.893,
    "relaxed_f1": 0.791, "slack_px": 3,
    "true_positives": 153828, "false_positives": 147255,
    "false_negatives": 1274997,
    "pred_fraction": 0.0335, "truth_fraction": 0.1588, "n_pixels": 9000000
  },
  "vector_stats": {
    "n_features": 2061, "total_length_km": 162.853,
    "mean_confidence": 0.892, "low_confidence_features": 33,
    "mask_fraction": 0.160427, "skeleton_pixels": 160596,
    "removed_small_objects": 784, "filled_holes": 12165,
    "crs": "EPSG:26986"
  },
  "n_stages": 26,
  "updated_at": "2026-09-08T11:33:12+00:00"
}
```

`artifacts` booleans drive which layer toggles are enabled — a layer whose
artifact is `false` should be visibly unavailable, not just broken when clicked.

**Endpoints available**

| method | path | purpose |
| --- | --- | --- |
| GET | `/api/health` | service + model status badges |
| GET | `/api/jobs` | list |
| POST | `/api/jobs` | create region |
| POST | `/api/jobs/upload` | user's own GeoTIFF |
| GET | `/api/jobs/{id}` | one job |
| DELETE | `/api/jobs/{id}` | remove |
| GET | `/api/jobs/{id}/previews` | rendered layer PNGs |
| GET | `/api/jobs/{id}/chips` | per-chip detail |
| GET | `/api/jobs/{id}/roads.geojson` | the deliverable |
| POST | `/api/jobs/{id}/pipeline` | run stages |
| POST | `/api/jobs/{id}/refine` | rework an area |
| GET/POST | `/api/jobs/{id}/chat` | talk to the agent |
| WS | `/ws/jobs/{id}` | live event stream |

**Geospatial constraints that affect layout**
- Regions are **square** (3000 × 3000 px for a 2×2 block). The map viewport
  should suit a square subject.
- Vectors are WGS84 GeoJSON — they drop into any web map.
- Imagery is EPSG:26986 (Massachusetts State Plane), reprojected for display.
- Resolution is 1 m/px: a residential street is only ~10–15 px wide. **Zoom
  affordances matter more than usual** — at default zoom, thin roads are nearly
  invisible, which is exactly where the AI makes its mistakes.

---

## 9. Visual direction

**Constraints (non-negotiable, technical):**
- No build step. Vanilla JS + CSS, Leaflet vendored locally. No React, no
  Tailwind CLI, no npm.
- Must work offline — no CDN fonts or assets.
- Dark UI is the current default and suits satellite imagery, which is dark and
  detailed. Light mode optional.

**Guidance:**

- **Let the imagery dominate.** Satellite photography is the content. Chrome
  should recede. The current three-column layout squeezes the map into a third
  of the screen — that's backwards.
- **Fix the overlay palette.** Prediction/truth/centreline/selection need to be
  distinguishable *on aerial imagery* (which is green, grey and brown) and
  colour-blind safe. Avoid red/green as the primary opposition. Consider
  magenta/cyan, which almost never occurs naturally in aerial photography.
- **Distinguish "agreement" from "layers".** There are two different overlay
  purposes: showing a layer, and showing correct-vs-missed-vs-false-positive.
  The second is a diagnostic view and probably deserves its own mode.
- **Numbers are the product.** Metrics shouldn't look like debug output. They're
  the evidence a professional stakes their work on.
- **Monospace for identifiers only** (job IDs, tile names, coordinates), never
  for prose.

Reference points worth borrowing from: Mapbox Studio and Felt for map-first
layout; Linear for dense information that stays calm; an observability tool like
Grafana Explore for streaming event traces.

---

## 10. Known problems with the current UI

Be direct about these — they're the reason for this brief.

1. **The map is too small.** It's the product, boxed into the middle third.
2. **Red/green overlays** are colour-blind hostile and semantically confusing.
3. **No progress during segmentation.** 1–2 minutes with only a status pill.
4. **Metrics have no interpretation.** Six bare decimals. A user cannot tell
   0.45 from 0.58 in significance, or know that 0.098 means "unusable".
5. **The agent trace is an undifferentiated stream.** 20–40 events with no
   hierarchy; the important ones (a score improving, a self-correction) look
   identical to routine ones.
6. **Layer toggles don't reflect availability.** A layer whose artifact doesn't
   exist yet still looks clickable.
7. **No first-run experience.** With zero jobs, the app is a set of empty panels.
8. **Refinement is buried.** Select-area-then-describe is the most novel
   interaction in the product and is currently a small button plus a text field.
9. **Mobile/narrow is unhandled.** Probably fine to stay desktop-only, but it
   should be a decision rather than an accident.

---

## 11. What success looks like

A GIS analyst opens the app, picks a region, and watches the AI trace the roads.
When it finishes they can tell **at a glance** whether the result is good, and
if it isn't, they can point at the bad part and say what's wrong in a sentence.
They export a GeoJSON they're willing to put their name on — and they know
precisely how much of it they checked.

The interface earns trust by showing its work, and by being honest when the
answer is bad.
