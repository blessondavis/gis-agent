# The agent harness, and finishing the last roads by hand

The model gets most of a road network. This document covers the two things
around the model that decide whether the rest gets done well: the **harness**
(the loop that lets an LLM drive the tools without drifting or overclaiming)
and the **editing layer** (the tools that let a person finish what the model
missed, in the same interface).

The harness borrows its structure from xAI's open-source
[Grok Build](https://github.com/xai-org/grok-build), adapted to a domain where
the artefacts are rasters and road networks rather than source files.

## What was borrowed, and how it changed

| Grok Build | here | why it had to change |
| --- | --- | --- |
| Typed stream events (`tool_call`, `tool_call_update`, `plan`) | `plan`, `tool_call` / `tool_result` paired by call id, `verify`, `checkpoint`, `edit` | The old UI matched results to calls by tool name, which breaks the first time a tool is called twice. |
| Plan mode: a state machine; only the plan file is writable | `mode="plan"` offers **read-only tools only**, and the UI shows **Approve & run** | Here the expensive, destructive step is re-running the model over a region, so the gate sits in front of that. |
| Todo list pinned in the TUI | `update_plan` harness tool, pinned above the chat, persisted per job | Handled by the harness, not the MCP server: a plan is about the conversation, not the pipeline. |
| Worktrees plus `worktree/apply` | `try_candidates` builds each setting in its own directory, and `apply_candidate` commits one | Settings are compared without touching the live result, and your edits are merged over every candidate. |
| `/goal` adversarial evidence review | **Stop gate**: the agent can't finish if the output changed since it last measured, or if its reply quotes a score no tool returned | The evidence is the tool results, so the check is deterministic: no second model is needed. |
| Oversized MCP results written to disk | Results over 6 KB are spilled to `spill/`; the model sees a compacted copy and the path | Previously they were blindly cut at 12,000 characters, mid-JSON. |
| Memory re-injected after compaction | **Working memory** rebuilt from the job's files before *every* model call | Best score so far, settings tried, your edits and the plan survive transcript trimming. |
| `/rewind` (conversation only; Grok's docs say files are *not* restored) | **Rewind restores the files**: mask, centrelines, confidence maps, edits and plan | A job is a handful of files, so it can. See [checkpoints](#checkpoints). |
| `grok -p` headless, streaming JSON | `gisagent agent <job> "..." [--plan] [--json]` | Same harness, for scripts and CI. |
| Session replay | The agent panel rebuilds from `events.jsonl` on reload, and a rewind truncates that log too | |

Not borrowed: parallel sub-agents. MCP calls go through one worker that owns
the only copy of the model on an 8 GB GPU, so parallel tool calls would just
queue. The parallelism is inside `try_candidates`, which scores its variants
on CPU threads.

## Judging candidates without labels: two ideas, one survived

`try_candidates` has to rank settings on imagery nobody has labelled. Both
candidate judges were tested before being built on, on the Boston region
across seven threshold and vectoriser variants, against the ground-truth
ranking:

| label-free judge | Spearman vs ground-truth quality |
| --- | --- |
| topology score (connectivity, dead ends) | **−0.64**: ranked the candidates backwards |
| expected F1 of the network under the model's confidence | **+0.96** (p < 0.001) |

Topology fails because pruning a network to its arterials makes it very
"coherent" and nearly useless. Expected F1 works by burning the centrelines
back in at label width (7 m) and treating the confidence map as calibrated
truth. Expected true positives are the confidence inside the footprint, and
expected road is all of it.

The same idea gives expected *precision* and *recall*, so the judge follows
what the person asked for. `objective` sets β in an F-β score:

| objective | β | expected vs true F-β | same winner as ground truth |
| --- | --- | --- | --- |
| balanced | 1 | +0.89 | yes |
| precision ("a wrong road is worse than a missing one") | 0.5 | +0.96 | yes |
| recall | 2 | +1.00 | yes |

Limits: this is one region and seven candidates. The judge is also
self-referential. It trusts the model's confidence, so it can't see a road the
model is confidently wrong about. It is fit for choosing *post-processing* for
a fixed model, which is its only use here, and not for judging the model.

This mattered in a live run. Asked for a base map where wrong roads are worse
than missing ones, the agent originally ranked by balanced quality. It applied
a candidate that *lowered* correctness (0.820 → 0.802) and reported "precision
rose", because pixel precision did. The objective parameter, plus one
paragraph of guidance, fixes the mismatch rather than the symptom.

## The editing layer

### Edits are a layer, not a modification

The agent regenerates `roads.geojson` every time it re-vectorises or reworks an
area. If edits went into that file, the next agent run would silently wipe the
person's work. So edits live in `edits.json`, an append-only log of
`add` / `delete` / `replace` / `dismiss` operations with undo and redo. They are
merged over the machine output into `network.geojson`, the deliverable, every
time the network is read.

Consequences worth knowing:

- **Deletions are recorded as geometry, not ids.** Machine features have no
  stable identity across re-vectorisation. A machine line is suppressed when
  60% of its length runs within 4 m of a deleted geometry. That still works
  after the agent re-traces the road a couple of metres off, and it keeps a
  crossing street that passes through the buffer for only a few metres.
- **Human work wins.** A machine line running along a human-drawn one is
  dropped as a duplicate.
- **Junctions are real.** A drawn road that ends on another is snapped to it,
  and that road is **split there**, so the graph has the junction and topology
  scoring sees a connected network, not a new dead end.
- **The agent and the person can work at the same time.** Nothing locks,
  because nothing they write overlaps.

### Tools

In the map toolbar: **Draw road** (D), **Reshape** (V: drag vertices, drag a
midpoint to add one, right-click to remove), **Delete** (X), **Review
suggestions** (R), plus undo/redo (Ctrl+Z / Ctrl+Shift+Z). Vertices snap to
existing junctions first, then to the nearest point along a road. There's no
drawing library; the editor is about 500 lines in `web/static/editor.js`.

### Suggestions, measured

Review mode steps through roads the network is probably missing. Each was
scored against ground truth on Boston (a suggestion counts as real if 60% of
its length is within 5 m of a labelled road):

| kind | what it is | real roads | top 20 |
| --- | --- | --- | --- |
| **gap** | two dead ends that nearly meet, where one continues toward the other (within 40°) | 56% | **80%** |
| **possible road** | a stretch the model was unsure about (confidence above 0.25 but below the network's cut-off) and nothing covers | 40% | 40–45% |

Gaps are ranked first and labelled "usually right". Possible roads are
labelled "check it", because about half are parking aisles and courtyards. A
"must connect to the network" filter was tried and rejected: it lifted
precision from 40% to 45% and threw away more than half the real ones.

### Measuring completion

Pixel IoU stops being the right question once a person is finishing the
network. A road drawn two pixels off-centre is still a finished road. The
**Completion** panel uses the length-based measures from the road-extraction
literature (Wiedemann): **completeness** (the share of real road length the
network covers) and **correctness** (the share of drawn length that is real),
within 5 m. It reports each **with and without your edits**, so the gap is
what the person's work added.

## Checkpoints

Each agent turn starts a checkpoint. The small artefacts (manifest, mask,
centrelines, network, edits, plan; about 2 MB) are copied then. The large
ones (per-chip confidence maps, the stitched confidence raster; tens of MB)
are copied only if and when the turn is about to overwrite them. Most turns
never re-run the model, so most checkpoints stay small.

The MCP server process does the overwriting, so the active checkpoint is a
file in the job directory rather than state in either process. Restoring an
older checkpoint unwinds the later ones newest-first, so every file ends at
its earliest recorded state.

The live end-to-end test checks this byte-for-byte. It runs plan-first, then
approves the plan; the agent runs `try_candidates` and `apply_candidate`; then
it rewinds. All 20 tracked files come back identical, including the 3 the turn
changed.

## Bugs the live runs found

Scripted tests pass on scripted models. These surfaced only with a real one:

- The model sent `mask_threshold` instead of `threshold`. It was silently
  ignored, so it produced five identical "candidates", and the model called
  the results confusing. The tool schema is now typed, known aliases are
  accepted, and any other unknown key is an error that names the valid ones.
- `close_radius: 0.5` gave an empty morphology footprint and a crash. All
  variant values are now coerced and range-checked.
- Reasoning models put their whole chain of thought in the message content,
  thousands of characters per step. The panel folds anything longer than a
  couple of sentences into a collapsed "reasoning" card.
- The step cards collapsed to 2 px. `#chatlog` is a scrolling flex column,
  and flex items with `overflow: hidden` have no minimum height. This was
  latent in the stylesheet and invisible until the log overflowed.

## Files

```
src/gisagent/agent/loop.py        the harness: plan tool, plan mode, stop gate, memory, spilling
src/gisagent/checkpoints.py       per-turn snapshots with copy-on-write for large artefacts
src/gisagent/vector/edits.py      edit log, merge, junction splitting, suggestions
src/gisagent/evaluate/network.py  completeness / correctness, expected precision / recall
web/static/editor.js              draw, reshape, delete, review; snapping; keyboard
tests/test_harness.py             the harness against a scripted fake model
tests/test_edits.py               merge rules, scoring, checkpoints, variant validation
```
