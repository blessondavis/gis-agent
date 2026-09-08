# Can a VLM judge annotation quality without ground truth?

The goal is to annotate **unlabelled** imagery. Ground truth is used here only
to check the machinery; on real work there is none, so something else has to
tell the loop whether an annotation is good enough to stop refining.

The proposed answer was a vision model as critic. That is a load-bearing
assumption — if the critic cannot tell a good annotation from a bad one, the
loop has no feedback signal and just spins. So it was tested before being built
on.

## Method

Six overlays of **known** quality were produced by degrading the ground truth in
ways that mirror failure modes actually observed in this project. Each was shown
to the model blind, in shuffled order, with no reference image and no hint of
which variant it was. The model returned JSON scores. Ground truth was used only
to build the variants and to check the answers afterwards.

| variant | what was done | why |
| --- | --- | --- |
| `perfect` | ground truth unchanged | ceiling |
| `missing_40` | 40 % of the area's roads erased | the dense-urban failure |
| `missing_75` | 75 % erased | severe version of the same |
| `spurious` | fake roads drawn across buildings | false positives |
| `shifted` | offset by 12 px | systematic misregistration |
| `blank` | nothing annotated | floor |

Erasure is **block-wise, not component-wise**. The first attempt dropped
connected components, which silently did nothing: in a city grid every street is
one connected component, so component dropping is all-or-nothing. Block-wise
erasure also matches the real failure — a model misses an *area*, not a random
scatter of segments.

Annotation is drawn in **magenta**, which essentially never occurs in aerial
photography, so the critic cannot confuse the overlay with terrain.

## Result

Model: `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning`, 512 px, temperature 0.1.

| variant | true IoU | VLM overall |
| --- | --- | --- |
| perfect | 1.000 | 78 |
| spurious | 0.915 | 55 |
| missing_40 | 0.636 | 70 |
| missing_75 | 0.247 | 30 |
| shifted | 0.127 | 50 |
| blank | 0.000 | 0 |

**Spearman rho = +0.886, p = 0.019** (n = 6). Pearson r = +0.808.

The critic ranks annotation quality reliably. The hypothesis holds.

## Where it fails, and what to do about it

Two findings matter more than the headline number.

**It cannot see misregistration.** `shifted` is objectively the second-worst
variant (IoU 0.127 — the annotation lies systematically off the roads) and the
critic scored it 50, middling. A uniform 12 px offset still *looks* like a road
network drawn over a road network. This is a real blind spot, and it is the
narrow point in the whole separation: the worst "good" annotation scored 55 and
the best "bad" one scored 50. Five points is not a margin to bet on.

**It penalises false positives more than IoU does.** `spurious` has IoU 0.915
but scored 55, below `missing_40` at IoU 0.636 scoring 70. The critic is
arguably *more right* than the metric here — roads drawn across buildings are a
worse defect than a few missing side streets, because they are wrong rather than
incomplete. Worth knowing that the critic's ordering is not the metric's
ordering, and that this is a feature.

## Consequence for the design

Do not use the VLM as the only signal. Use it as one of three, because the other
two are free and cover exactly what it misses:

| signal | catches | cost |
| --- | --- | --- |
| **VLM critic** | missing roads, roads drawn over buildings | an API call |
| **Topology** (QGIS) | fragmentation, dangles, gaps — and misregistration, indirectly | deterministic, local |
| **Model confidence** | the model's own uncertainty | already computed |

Topology is the important addition. A real road network is *connected*: streets
meet at junctions, dead ends are rare, and the graph is one component, not two
hundred. `grass:v.net.connectivity` and `native:checkgeometrydangle` measure
that with no labels and no model, and a badly misregistered annotation degrades
its topology in ways a uniform shift does not hide.

So the stop condition for a refinement loop on unlabelled imagery should be a
combination, not the VLM alone — and when the VLM and the topology disagree,
that disagreement is itself the signal to surface to a human.

## Reproducing

`scratchpad/vlm_critic_test.py` (not shipped — it is an experiment, not a
feature). Notes for anyone repeating it:

- NIM stalls on large inline images. A 640 px JPEG at quality 85 is ~259 KB of
  base64 and times out consistently; 512 px at quality 72 is ~115 KB and
  answers in ~30 s.
- `moonshotai/kimi-k3` reads aerial imagery best of the models tested, but is
  rate-limited hard on a shared key, and it spends its token budget on
  `reasoning_content` first — `max_tokens=200` returns an empty string. Give it
  2000.
- `nvidia/cosmos-reason2-8b` is in the NIM catalogue but returns 404 "not found
  for account" without a specific entitlement. So do `vila`, `neva`,
  `phi-3-vision` and `kimi-k2.6`.
- `google/gemma-4-31b-it` reads the imagery correctly but timed out on every
  scored call.
