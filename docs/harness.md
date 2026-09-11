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

## Autonomous mode: rules the harness enforces

`gisagent annotate <job>` (or **Annotate autonomously** in the web app) hands
the agent a whole region and walks away. Prompt text alone doesn't hold up for
that. A model will skip a step, repeat a failing call, stop at the first
plausible result, or finish on a number it misremembered. So every rule in
[`agent/rules.py`](../src/gisagent/agent/rules.py) exists twice: once as text
the model reads, and once as code the harness runs on every call and before
every attempt to finish.

| rule | enforced by |
| --- | --- |
| **Order.** tile → segment → stitch → vectorise → candidates → improve → hand off | A call whose prerequisites are missing is refused with the step that is actually next, before any GPU time is spent. `evaluate_result` on unlabelled imagery is refused with what to use instead. |
| **The harness measures.** | After every change to the network, the harness scores it itself under the task's objective. It uses ground truth when there is some, else expected precision and recall under the model's confidence. The score is appended to the tool result, so the model never has to remember to measure. |
| **Keep the best.** | The best-scoring result is snapshotted. The agent can't finish below it, and `restore_best` brings it back. In conversation mode too. After a model re-run, the new network is judged under the best result's confidence map (see below). |
| **Measured domain facts.** | Upscaled U-Net runs are refused (they hurt every time they were measured), and so are native-scale U-Net area reworks, which can't change anything. The refusal carries the evidence and the lever to use instead. |
| **Scope.** | A conversation is bound to its job. The harness fills in `job_id` on every call, and creating another region is refused. |
| **The task's objective wins.** | `try_candidates` is always run with the task's objective, whatever the model typed. `set_objective` changes it explicitly and re-scores the history. |
| **Budgets.** | Whole-region model runs (2), area reworks (4) and candidate rounds (3) per task. A call over budget is refused. |
| **No loops.** | The same call with the same arguments *on the same job state* is refused. Vectorising again after a rework is fine; repeating it on an unchanged mask is not. |
| **A stop condition decided by code.** | Targets met (labelled jobs), a **plateau** (two changes in a row with no new best), or **no productive lever left**. Once plateaued, further improvement calls are refused. An agent sent back twice without acting is stopped, and the report says what was left undone. |
| **Definition of done.** | A measured network exists. Its settings were chosen with `try_candidates` on the *current* confidence map, and the live result is at least as good as the best candidate. It isn't below the best measured result. A stop condition holds. `suggest_missing_roads` has run on the final network. Finishing is blocked, up to eight times, until all of these hold. |
| **A report from the checks.** | `report.json` / `report.md`: a status (complete / best effort / incomplete / failed) with its reason, the score after every change, the budgets used and the review queue left for a person. The verdict is computed, not taken from the model's summary. |

One subtlety shaped the design. Label-free scores are only comparable
**within one confidence map**, because re-running the model changes the map
the judge scores against. Each measurement is tagged with the map's version,
hashed from its content rather than its timestamp, because `apply_candidate`
rewrites an identical file. When the version changes, the new network is
judged **under the best result's map**, the one biased against it. Only a
network that wins even there becomes the new best. This was the third version
of the rule, and the only one that survived measurement (see below). The
plateau is "two changes in a row with no new best", which means the same thing
with or without labels.

### Unattended runs

Each run started from a bare mosaic with no human input, using the live LLM:

| run | objective | what the agent did | result |
| --- | --- | --- | --- |
| Andover, labelled (CLI) | balanced | ran the pipeline, applied a candidate, reworked an area, ran a second candidate round, hit a plateau, handed off | 0.848 → 0.853 against ground truth; **best effort (plateau)** |
| Boston, labels hidden (CLI) | precision | pipeline, then three candidate rounds varying the vectoriser; sent back once to use its last round | expected score 0.815 → 0.846; **best effort (no useful budget left)**; 30 gaps handed off |
| Andover, labelled (web button) | precision | the same, from **Annotate autonomously** | 5 measured changes; **best effort (plateau)**, in 3 min 42 s |

**Checked against the hidden truth**, the unlabelled Boston run beat the default
pipeline on both axes. Correctness went from 0.837 to 0.851, completeness from
0.654 to 0.667, and the precision objective (F0.5) from 0.793 to 0.807. A second
run reproduced this to three decimals. The agent never saw a label: it judged
every change with the expected-score judge alone.

### What the unattended runs taught the rules

Every run broke a rule that had looked sound on paper. Each fix is covered by a
test that replays what happened.

1. **The job ID.** The model mistyped a 22-character ID, and the loop rule
   refused its corrected retry. The harness now fills in `job_id` itself.
2. **Keep-best across model re-runs, twice.** Label-free scores depend on the
   confidence map, and re-running the model changes it. The first rule started
   a fresh comparison series; the second re-scored the old best under the *new*
   map. Both let a worse rework replace a better network. The fix came from an
   experiment: eight area reworks on Boston, all of which truly hurt. Judged
   under the new map, the rule said "keep" 8 times out of 8. Judged under the
   *best result's* map, which is biased against the newcomer, it said "reject"
   8 times out of 8. That is the rule now. Caveat: no rework in the
   experiment helped, so its ability to accept a good one is untested.
3. **Upscaling the U-Net hurts.** Those eight reworks used `refine_area`'s
   default of upscale 2, inherited from SAM 3, where it helped. With the U-Net
   they cost up to 0.36 F0.5, and at native scale a rework reproduces the same
   prediction. Both are now refused with the evidence, and the playbook points
   "improve" at candidate rounds, the lever that actually worked.
4. **Rules must be consistent.** "Budget spent" required every budget to be
   exhausted, including the ones rule 3 had made unusable, so no stop condition
   could hold. The harness sent the agent back eight times. The stop condition
   is now "no productive lever left", and an agent that stops acting after
   being sent back is stopped rather than sent back forever.
5. **Scope.** An agent created a second region from other tiles. Job binding
   kept its work on the right job, but an orphan was left behind. A bound
   conversation can no longer create regions.

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
- **Autonomous run, labelled.** The model mistyped the job ID (`…f00400438` for
  `…f00438`), and the loop rule then refused its corrected retry, because the
  ID wasn't part of the repeat key. A conversation is bound to one job, so the
  harness now fills in `job_id` itself on every call.
- **Autonomous run, unlabelled.** A rework changed the confidence map. Under
  the first keep-best rule, that started a new comparison series, so the
  worse result became "best" and overwrote the snapshot of a better one. The
  better network was unrecoverable. The best is now re-scored under the
  current map, and a regression test replays the exact sequence.
- The step cards collapsed to 2 px. `#chatlog` is a scrolling flex column,
  and flex items with `overflow: hidden` have no minimum height. This was
  latent in the stylesheet and invisible until the log overflowed.

## Files

```
src/gisagent/agent/loop.py        the harness: plan tool, plan mode, stop gate, memory, spilling
src/gisagent/agent/rules.py       the task rules: order, budgets, loops, measuring, keep-best, done, report
src/gisagent/checkpoints.py       per-turn snapshots with copy-on-write for large artefacts
src/gisagent/vector/edits.py      edit log, merge, junction splitting, suggestions
src/gisagent/evaluate/network.py  completeness / correctness, expected precision / recall
web/static/editor.js              draw, reshape, delete, review; snapping; keyboard
tests/test_harness.py             the harness against a scripted fake model
tests/test_edits.py               merge rules, scoring, checkpoints, variant validation
tests/test_rules.py               every task rule, and a whole scripted autonomous task
```
