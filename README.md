# Time-Budget-VLN

Fork of [NaVILA](https://github.com/AnjieCheng/NaVILA), extended with **V1: budget-conditioned
multi-goal navigation** — teaching NaVILA to change its plan based on a time/step budget.
Same scene, same two goals: a tight budget reaches the nearest goal only; a loose budget
visits both. The signature result is a behavior *flip*: identical scene and goal pair, only
the budget number changes, and the path qualitatively differs.

## Signature behavior flip

Same scene (`GdvgFV5R1Z5`), same two goals (sink, toilet). Left: tight budget reaches the
nearer goal only. Right: loose budget visits both.

<p align="center">
  <img src="multigoal_episodes/gifs/ep0_near_GdvgFV5R1Z5.gif" width="48%">
  <img src="multigoal_episodes/gifs/ep0_both_GdvgFV5R1Z5.gif" width="48%">
</p>

Each clip is [RGB camera | depth | top-down map with trajectory and goal markers], rendered
from a privileged shortest-path expert (not a trained model) — see
[`multigoal_episodes/`](multigoal_episodes/) for the generation code.

## Status

**Phase 1 (done):** budget-conditioned multi-goal episode generation — sampling goal pairs
on MP3D scenes and computing empirical nearest-only / both-goal step costs with a geodesic
expert. See [`multigoal_episodes/README.md`](multigoal_episodes/README.md).

**Next:** turn episodes into budget-conditioned instructions + expert action sequences
(SFT training data), then LoRA fine-tune NaVILA's high-level VLM.

## Base model

This repo builds on NaVILA's training/eval pipeline (Habitat + MP3D, VLN-CE). For the
original installation, dataset, training, and evaluation instructions, see
[`NAVILA_UPSTREAM_README.md`](NAVILA_UPSTREAM_README.md).
