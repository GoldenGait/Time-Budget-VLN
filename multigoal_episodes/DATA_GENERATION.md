# Budget-VLN Data Generation — Understanding / Handoff

*What the datasets under `multigoal_episodes/` are, why they exist, and how they're built.*

---

## 0. The one-sentence thesis

We are teaching NaVILA a **behavior flip conditioned on a stated budget**: given the *same*
scene, start pose, and target, the agent should behave **differently** depending on how much
"budget" (in primitive-step units) the instruction grants it. Small budget → go straight to the
goal. Large budget → you can afford to wander / cover more before committing. The dataset is
built so that **no global budget threshold** separates the two behaviors — the model must read
the budget *relative to the scene*, not memorize a cutoff.

Everything below is machinery in service of producing that contrastive supervision.

---

## 1. Two-layer architecture (applies to every dataset here)

All datasets share the same pipeline shape:

```
Layer 1   generate_*.py        oracle/expert rollout in Habitat-sim
          episodes_<scene>.json   → poses + stored primitive action TRACES (no pixels)

Layer 1.5 assign_budgets.py     attach a per-episode budget number to each regime
                                  (decorrelated so no global cutoff works)

Layer 2   build_sft_dataset.py  replay each trace, DUMP egocentric 512×512 frames,
                                  aggregate primitives → NaVILA vocab, emit annotations.json
                                  → egocentric frames + {video_id,q,a,frames}  ← what NaVILA trains on
```

Key invariants:
- **Primitive action unit** everywhere: `0.25 m` forward, `15°` turn. Success radius `0.2 m`.
- **Layer 1 stores traces, not frames.** A trace is a list of primitives. Frames are only
  rendered at Layer 2, so Layer 1 is cheap and (for exploration) CPU-only.
- **Layer 2 aggregates** primitives into NaVILA's native vocabulary — `move forward 25/50/75 cm`,
  `turn left/right 15/30/45°`, `stop` — capping 3 primitives per emitted action.
- **The live remaining budget is re-injected into the instruction at every step.**

> ⚠️ Layer 2 output (frames + `annotations.json`) is written to the external HDD
> `/media/maitree-tiamat/Expansion/NaVILA_data/budget_vln_sft/`, **not** into git. Only Layer 1
> JSON + the scripts live in the repo. The visualization `.mp4`s in `videos/` are for eyeballing
> experts and are **not training data**.

---

## 2. Dataset A — Shortest-distance (multi-goal): `tight` vs `loose`

**Generator:** `scripts/generate_episodes.py` · **Trace keys:** `traces.tight`, `traces.loose`

The original budget signal. For each MP3D scene we sample `(start, G1, G2)` — a start pose and
**two** goal objects from a whitelist — and use habitat-sim's `GreedyGeodesicFollower` (a geodesic
shortest-path oracle) to measure the empirical cost, in primitive steps, of:

| Field                | Meaning |
|----------------------|---------|
| `segment_costs.S_G1` | shortest cost start → G1 |
| `segment_costs.S_G2` | shortest cost start → G2 |
| `nearest_only`       | `min(S_G1, S_G2)` — reach the **nearer** goal and stop |
| `both_tour`          | cheaper of the two 2-leg orderings that visit **both** goals |
| `ordering_for_both`  | which order (G1→G2 or G2→G1) achieves `both_tour` |

Two expert traces are stored:
- **`tight`** — navigate to the nearest goal, then `stop`. Cost `≈ nearest_only`.
- **`loose`** — visit *both* goals, then `stop`. Cost `≈ both_tour`.

**Behavior flip:** a `tight` budget lands in `[nearest_only, both_tour)`; a `loose` budget is
`≥ both_tour`. Same scene + same goal pair — **only the budget number and which trace we replay
differ**. That's the supervised flip.

**Filter:** an episode is kept only if `both_tour / nearest_only ≥ 1.5`, so there's enough room
for a budget to straddle the flip point (otherwise the two behaviors are indistinguishable).

**Note:** `tight`/`loose` here use **no vision** — they navigate to *known* goal navpoints and
stop at geodesic ≤ 0.2 m. Vision only matters for exploration (Dataset B).

---

## 3. Dataset B — Frontier exploration (object-search): the "infinite-budget" extreme

The loose regime above still *knows both goal locations*. Dataset B pushes the loose end to its
limit: an **object-search** task where the agent is told **only the target word** and must
**explore target-agnostically until the object comes into view**, then stop.

It **reuses each tight episode's** `(scene, start, tight-target)` so that shortest-path (`tight`)
and exploration form a **contrastive pair over an identical setup** — the only thing that changes
is the strategy (and therefore the length, and therefore the required budget).

**The stop condition is the crux, and it is GEOMETRIC, not pixel-based.** A semantic-sensor pixel
test fails on low furniture (a bed/stool/chest renders ~0 px in a horizontal 1.25 m camera — the
agent looks *over* it; only tall objects like showers work). So "found" =
- target within **`SEE_R = 2.0 m`** geodesic, **AND**
- its center inside the forward **half-FOV cone (45°)**, **AND**
- **line-of-sight clear** (navmesh sampling; LoS skipped under 1.0 m so the object's own footprint
  doesn't false-occlude).

The target location is used **only** in this stop check — the *path* stays target-agnostic. The
big consequence: **Layer 1 needs no camera → pure CPU, ~1–2 s/episode**, embarrassingly parallel
across the 86 scene files. GPU is only needed at Layer 2 (frame dump).

We explored **two different exploration strategies** for this:

### B1 — `explore`: emergent frontier-coverage search  ✅ committed
**Generator:** `scripts/generate_frontier_episodes.py` · **Trace key:** `traces.explore`

Classic frontier exploration — the agent builds a map as it goes and chases the boundary of the
unknown:
- Occupancy grid from `pathfinder.get_topdown_view(0.5, floor_y)` (fast; an earlier per-cell
  `snap_point` version hung).
- **FOV-gated reveal** (`REVEAL_R = 2.0 ≤ SEE_R`, so nothing gets revealed-away without a
  detectable look).
- **Frontiers** = free cells 8-adjacent to unknown-navigable; pick the nearest frontier-cluster
  centroid by geodesic; walk the leg with `GreedyGeodesicFollower`.
- **360° look-around scans** at start, at each frontier arrival, and every 12 forward steps
  mid-leg — otherwise the follower flies past side objects without ever "seeing" them.
- No behavioral step cap; safety valve at 3000 primitives.

**Smoke results (scene 17DRP5sb8fy, 20 eps):** 95% found; explore/tight length ratio median ≈16×
(good budget contrast). Discard degenerates (`explore = 0`, i.e. target visible at spawn) and the
~5% give-ups. Video: `videos/ep{0,6}_explore_17DRP5sb8fy.mp4`.

*Character:* emergent, greedy, human-like wandering. Length is variable and can be long. No
completeness guarantee (the ~5% give-ups).

### B2 — `covtour`: coverage-tour over precomputed viewpoints  🧪 prototype only
**Prototype:** `scratchpad/coverage_tour_demo.py` (not yet promoted to repo) · would be
`traces.covtour`

Instead of *emergent* frontier search, precompute a route that is **guaranteed** to see
everything:
- **Greedy set-cover:** find the fewest viewpoints whose `VIS_R = 2.0 m` + LoS discs cover *all*
  navigable cells.
- **Nearest-neighbour tour:** order those viewpoints into a short tour from the start pose
  (geodesic), then walk it.
- Same geometric stop check; turn to face the target before stopping.
- Target-agnostic: the tour is built **without** the target; the target enters only at the stop
  check.

*Character:* **bounded-length, complete coverage → reliably finds any reachable target** (no
give-ups). More systematic / less human-like than B1. Video:
`videos/ep{0,6}_covtour_17DRP5sb8fy.mp4`.

### explore vs covtour — the trade-off we were weighing

| | **B1 `explore`** (frontier) | **B2 `covtour`** (coverage tour) |
|---|---|---|
| Route | emergent, map-as-you-go | precomputed set-cover + NN tour |
| Completeness | none (~5% give-ups) | guaranteed complete coverage |
| Length | variable, can be long | bounded |
| Realism | human-like wandering | systematic sweep |
| Status | **committed** generator | scratchpad prototype |

Both are valid "loose extreme" experts; the open question is which produces better contrastive
supervision (and whether the systematic covtour is *too* legible a pattern for the model to
shortcut).

---

## 4. Layer 1.5 — Budget assignment (why no global cutoff works)

**Script:** `scripts/assign_budgets.py`

Given a kept episode with its own flip point `explore_len` (and feasible `tight_len`), assign a
**narrow band straddling that episode's own flip point**, in primitive-step units:

```
explore budget = explore_len * (1 + U[gap, span])     # just above → search feasible
tight   budget = explore_len * (1 - U[gap, span])     # just below → explore infeasible,
                 clamped up to tight_len so the tight trace stays feasible
```

Because both budgets **hug `explore_len`, which varies widely across episodes**, their marginal
distributions **overlap** — a big-`explore_len` episode's *tight* budget can exceed a small
episode's *explore* budget. So **no single global threshold K** separates the two behaviors; the
model is forced to interpret the budget **relative to the scene**.

- A "wide, explore-up-to-a-shared-ceiling" scheme was tried and **failed** this guardrail (explore
  budgets skewed high, tight low → a global cutoff separated them).
- Keep an episode only if the explore expert **found** the target **and** `explore_len > tight_len`
  (drops give-ups and found-at-spawn degenerates). **Never touches `test_heldout`.**
- Same guardrail logic as `autoresearch/build_contrastive_pairs.py`.

---

## 5. Layer 2 — What NaVILA actually consumes

**Script:** `scripts/build_sft_dataset.py`

For each episode × regime, replay the stored trace and:
1. Dump one **egocentric 512×512 RGB frame** per action decision point (first-person camera at
   1.25 m — *not* the top-down maps).
2. Aggregate `0.25 m / 15°` primitives → NaVILA native actions (cap 3 prims/action).
3. Write the budget-conditioned instruction with the **live remaining budget** at that step.
4. Emit one `{video_id, q, a, frames}` record per action, plus a final `stop`.

Output mirrors the released NaVILA-Dataset layout so `llava/data/datasets_mixture.py` can point
at it directly:
```
<out>/annotations.json                 (data_path)
<out>/frames/<video>/frame_K.jpg       (image_path = <out>/frames)
```

**So: NaVILA trains on the egocentric frames + `annotations.json`, never on the `episodes_*.json`
poses directly and never on the top-down `.mp4` visualizations.**

---

## 6. Totals & status

**Layer 1 corpus:** 86 scene files, **1,720 episodes** = 1,200 train / 220 val / 300
test_heldout (whole-scene holdout).

| Piece | Script | In git? | Notes |
|---|---|---|---|
| Shortest-dist `tight`/`loose` | `generate_episodes.py` | ✅ | Dataset A, complete |
| Frontier `explore` | `generate_frontier_episodes.py` | ⚠️ untracked | B1, smoke-validated, needs scale-out |
| Coverage-tour `covtour` | `scratchpad/coverage_tour_demo.py` | ❌ prototype | B2, not promoted |
| Budget assignment | `assign_budgets.py` | ⚠️ untracked | Layer 1.5 |
| SFT flattener | `build_sft_dataset.py` | ✅ | Layer 2 |
| Egocentric frames + annotations | — (build output) | ❌ external HDD | the actual training data |
| Top-down `.mp4`s | `render_episode_video.py` etc. | partial | inspection only, not training |

**Open TODO (from smoke → production):**
- Promote `explore` smoke → repo generator writing `traces.explore` (done as
  `generate_frontier_episodes.py`; still needs to be **run across all 86 scenes**).
- Decide **explore vs covtour** (or keep both as separate loose experts).
- Run decorrelated `assign_budgets.py` over the full set; re-audit the global-cutoff guardrail.
- Quantitative leak audit (early-bearing vs target-bearing correlation ≈ 0).
- Run Layer 2 across all scenes and confirm frames exist on the Expansion drive before training.
