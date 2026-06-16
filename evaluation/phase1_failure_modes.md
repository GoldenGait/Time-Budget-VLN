# Phase 1 — Behavioral Mode Breakdown (goal_only)

Per-episode behavior classification for the two `goal_only` cells, derived from the
logged trajectories. Reproducible via [classify_phase1_modes.py](classify_phase1_modes.py):

```bash
cd ~/NaVILA/evaluation && conda activate navila
python classify_phase1_modes.py --input phase1_results.json
```

## Modes

| Mode | Definition |
|---|---|
| **SUCCESS** | reached the goal |
| **WANDER(cover)** | hit the 500-step cap, moved around but never recognized/stopped |
| **WANDER(revisit)** | hit the cap, churning the same ground (path ≫ covered area) |
| **SPIN** | turned in place (>85% of moves are turns, ~no displacement) |
| **EARLY_QUIT** | issued STOP but ended >3 m from goal |

Geometry (x,z plane): `disp` = straight-line start→final, `path` = cumulative path length,
`detour` = path/disp (1.0 = straight), `cov` = distinct 0.5 m cells, `turn%` = fraction of moves that are turns.

## `phase1_baseline / goal_only` (unchanged scaffold)

| ep | mode | steps | disp | path | detour | cov | turn% | NE |
|---|---|---|---|---|---|---|---|---|
| 53 | SUCCESS | 78 | 11.1 | 13.9 | 1.3 | 31 | 0.27 | 2.6 |
| 63 | SUCCESS | 67 | 9.2 | 12.6 | 1.4 | 28 | 0.23 | 2.7 |
| 464 | SUCCESS | 33 | 3.2 | 3.8 | 1.2 | 8 | 0.53 | 1.9 |
| 1088 | SUCCESS | 28 | 5.2 | 5.2 | 1.0 | 11 | 0.22 | 0.4 |
| 1388 | SUCCESS | 34 | 5.8 | 5.9 | 1.0 | 14 | 0.27 | 2.1 |
| 1683 | SUCCESS | 188 | 8.0 | 11.2 | 1.4 | 25 | 0.76 | 0.3 |
| 60 | EARLY_QUIT | 39 | 3.5 | 5.0 | 1.4 | 10 | 0.47 | 4.3 |
| 200 | EARLY_QUIT | 57 | 1.9 | 6.2 | 3.2 | 14 | 0.55 | 6.1 |
| 431 | EARLY_QUIT | 22 | 2.0 | 2.0 | 1.0 | 5 | 0.57 | 7.9 |
| 507 | EARLY_QUIT | 30 | 6.0 | 6.0 | 1.0 | 14 | 0.17 | 12.2 |
| 777 | EARLY_QUIT | 52 | 9.5 | 9.9 | 1.1 | 20 | 0.18 | 19.7 |
| 1254 | EARLY_QUIT | 315 | 4.8 | 5.6 | 1.2 | 14 | 0.93 | 12.4 |
| 1389 | EARLY_QUIT | 33 | 5.5 | 5.5 | 1.0 | 12 | 0.31 | 9.6 |
| 190 | WANDER(cover) | 500 | 3.4 | 23.0 | 6.7 | 40 | 0.81 | 7.5 |
| 288 | WANDER(cover) | 500 | 3.7 | 3.8 | 1.0 | 9 | 0.01 | 5.8 |
| 1187 | WANDER(cover) | 500 | 3.9 | 3.9 | 1.0 | 8 | 0.96 | 5.6 |
| 1378 | SPIN | 16 | 0.0 | 0.0 | 0.0 | 1 | 1.00 | 5.5 |

**Mode counts:** SUCCESS 6 · EARLY_QUIT 7 · WANDER(cover) 3 · SPIN 1

## `phase1_explore / goal_only` (exploration-nudged scaffold)

| ep | mode | steps | disp | path | detour | cov | turn% | NE |
|---|---|---|---|---|---|---|---|---|
| 60 | SUCCESS | 59 | 5.4 | 8.5 | 1.6 | 20 | 0.41 | 1.9 |
| 200 | SUCCESS | 54 | 4.8 | 5.2 | 1.1 | 11 | 0.60 | 2.5 |
| 1088 | SUCCESS | 28 | 5.2 | 5.2 | 1.0 | 11 | 0.22 | 0.4 |
| 1187 | SUCCESS | 128 | 7.7 | 8.6 | 1.1 | 18 | 0.71 | 1.7 |
| 1388 | SUCCESS | 52 | 5.8 | 8.2 | 1.4 | 19 | 0.35 | 0.7 |
| 1683 | SUCCESS | 416 | 7.8 | 11.3 | 1.4 | 26 | 0.89 | 0.6 |
| 53 | EARLY_QUIT | 57 | 9.7 | 10.7 | 1.1 | 25 | 0.23 | 5.1 |
| 63 | EARLY_QUIT | 31 | 1.6 | 2.3 | 1.4 | 7 | 0.70 | 8.1 |
| 507 | EARLY_QUIT | 39 | 5.6 | 6.5 | 1.2 | 14 | 0.32 | 6.6 |
| 777 | EARLY_QUIT | 172 | 10.6 | 27.8 | 2.6 | 39 | 0.32 | 23.0 |
| 1389 | EARLY_QUIT | 304 | 6.5 | 8.5 | 1.3 | 18 | 0.89 | 10.8 |
| 190 | WANDER(cover) | 500 | 22.6 | 52.5 | 2.3 | 84 | 0.58 | 33.7 |
| 288 | WANDER(cover) | 500 | 3.7 | 3.8 | 1.0 | 9 | 0.01 | 5.8 |
| 431 | WANDER(cover) | 500 | 9.2 | 10.9 | 1.2 | 24 | 0.02 | 11.8 |
| 464 | WANDER(cover) | 500 | 1.5 | 1.5 | 1.0 | 4 | 0.99 | 4.2 |
| 1254 | WANDER(cover) | 500 | 3.9 | 4.6 | 1.2 | 13 | 0.96 | 14.5 |
| 1378 | WANDER(cover) | 500 | 0.0 | 0.0 | 0.0 | 1 | 1.00 | 5.5 |

**Mode counts:** SUCCESS 6 · EARLY_QUIT 5 · WANDER(cover) 6

## Key finding: the explore scaffold *reshuffles* successes, it doesn't add them

Net SR is identical (6/17 → 6/17), but the *set* of solved episodes changes:

| | episodes |
|---|---|
| baseline successes | 53, 63, 464, 1088, 1388, 1683 |
| explore successes | 60, 200, 1088, 1187, 1388, 1683 |
| **gained** by explore | **60, 200, 1187** |
| **lost** by explore | **53, 63, 464** |

- **Rescued (gained):** 60 and 200 were EARLY_QUITs and 1187 a cap-wanderer under baseline — the "keep exploring, don't stop until the target is clearly visible" nudge pushed them to keep going and find the goal.
- **Broken (lost):** 53, 63, 464 were *clean short approaches* under baseline — the same nudge made them over-explore past the goal and either quit far away (53, 63) or wander to the cap (464).

So the exploration prompt **helps episodes that need exploration but harms episodes that just need to approach** — the two skills pull in opposite directions, and a single prompt can't serve both. It also shifts the failure profile from SPIN/EARLY_QUIT toward WANDER(cover): more genuine motion, but still no recognition/termination, so it just times out (cap hits 3 → 6).

## Takeaway

This reinforces the §10 conclusion of [phase1_design.md](phase1_design.md): the approach-vs-explore split is real and **prompting trades one for the other rather than fixing either**. A model that both explores *and* terminates correctly has to be *trained*, not prompted — exactly the budgeted-search fine-tuning direction. These 17 episodes, with their per-episode mode labels, are a ready before/after probe: the goal is to convert EARLY_QUIT/WANDER episodes to SUCCESS **without** regressing the clean-approach successes.

---

## Stop-clause disentangling (run 2026-06-11)

The `phase1_explore` pipeline changed **two** things vs baseline at once — the framing ("actively explore…") *and* the stop clause ("stop only once the target is clearly visible" vs baseline's "stop if the task is completed"). The rescue/break story could therefore be a **stop-condition artifact** rather than a framing effect. To disentangle, we ran a third cell holding the action-menu identical:

- **`phase1_explore_origstop`** = explore framing **+ the original** "stop if the task is completed" clause (factorial: explore-framing × original-stop).

Reproduce: `python run_phase1.py --pipelines phase1_explore_origstop --conditions goal_only`. The scaffold refactor in [run_phase1.py](run_phase1.py) asserts the composed `phase1_baseline`/`phase1_explore` strings are byte-identical to the originally-run versions, so the prior 68 records remain valid.

### Attribution — diagnostic episodes across all three cells

| ep | baseline | explore | explore_origstop | |
|---|---|---|---|---|
| **53** (break) | SUCCESS | EARLY_QUIT | **EARLY_QUIT** | stays broken |
| **63** (break) | SUCCESS | EARLY_QUIT | **EARLY_QUIT** | stays broken |
| **464** (break) | SUCCESS | WANDER(cover) | **WANDER(cover)** | stays broken |
| **60** (rescue) | EARLY_QUIT | SUCCESS | **SUCCESS** | persists |
| **200** (rescue) | EARLY_QUIT | SUCCESS | **SUCCESS** | persists |
| **1187** (rescue) | WANDER(cover) | SUCCESS | **SUCCESS** | persists |

Mode counts `explore_origstop`: SUCCESS 5 · EARLY_QUIT 5 · WANDER(cover) 7. (SR 5/17 vs 6/17 for both baseline and explore.)

### Finding: the antagonism is a **framing** effect, not a stop-clause artifact

Reverting the stop clause to the original **recovered none of the broken approaches** (53/63/464 stay broken) and **kept all three rescues** (60/200/1187 still succeed). Both halves of the rescue/break tradeoff are driven by the **exploration framing itself**. The methodological flag is resolved: the antagonism claim is airtight, not a stop-condition artifact.

**Secondary nuance (the stop clause helped, mildly):** episode **1683** (the productive long search) *succeeded* under explore's strict "target visible" stop (terminated at 416 steps) but **wandered to the cap** under explore_origstop's lenient stop — so the strict stop clause aided recognition-termination in that case, the *opposite* of the confound hypothesis. This is why explore_origstop nets one fewer success (5 vs 6) than explore: same framing-driven rescues/breaks, minus 1683 which the lenient stop let drift.

**Implication for fine-tuning:** framing controls *whether it explores vs approaches* (and these conflict); the stop clause is a smaller, somewhat-helpful knob on *termination*. Neither can be set to win both skills at once by prompt — confirming that explore-and-terminate must be **trained**. The four-cell factorial (`phase1_baseline_tgtstop` is registered but unrun) could close the loop on the stop-clause main effect if needed.
