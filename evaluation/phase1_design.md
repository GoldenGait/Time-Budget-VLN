# Phase 1 — Goal-Only Search Probe for NaVILA

**Goal:** Test, zero-shot, whether NaVILA can perform *goal-only search* — given a destination/landmark but **no route** — instead of the step-by-step route-following it was trained on. No fine-tuning, no model changes.

**Date:** 2026-06-11
**Researcher:** Maitree (Relling Systems)
**Model:** `navila-llama3-8b-8f` (LLaMA-3 8B + SigLIP, 8 video frames) — checkpoint at `~/models/navila-llama3-8b-8f`
**Simulator:** Habitat-Sim via VLN-CE, R2R-CE `val_unseen` (MP3D scenes)
**Runtime env:** conda `navila` (py3.10, torch 2.7.0+cu128, habitat 0.2.1, habitat_sim 0.2.3), GPU: RTX 5090

---

## 1. Research question

NaVILA is trained on **VLN** (R2R/RxR): dense, turn-by-turn route instructions ("exit the living room, turn right into the kitchen, …, wait in the room across the hallway"). The goal is implicit in following the route; success = stop within 3 m of the route's endpoint.

**"Find a mug" / "go to the kitchen" is a different task family (ObjectNav-like):** the instruction is sparse (a goal/landmark, no route), and reaching it requires *search* — exploring unseen space, using semantic priors, recognizing the target. We want to know whether the route-follower has any latent search ability before deciding whether to keep NaVILA or switch models.

### Search decomposes into two skills
- **Approach** — target already in view → walk to it and stop. NaVILA likely *has* this: VLN instructions already include object-grounded stops ("stop next to the treadmill", "the room with a clock") and a learned STOP.
- **Explore** — target not in view → systematically uncover unseen space (head to rooms where the target likely is; cover ground; avoid revisiting). NaVILA was **never trained or rewarded for this**, and has no map + only 8 frames of memory.

Prediction: NaVILA approaches well, explores poorly. The experiment is designed to expose this split rather than give a single pass/fail.

---

## 2. Design

We reuse the existing R2R-CE benchmark with **zero new data** by *stripping the route* from each instruction and keeping only the goal/landmark. Same scenes, same goal coordinate, **same success metric (3 m geodesic to goal)** — so the only variable removed is the route.

### Two instruction conditions (the independent variable: route vs. no route)
| Condition | Instruction | Tests |
|---|---|---|
| **`original`** | unmodified route instruction (control) | route-following baseline |
| **`goal_only`** | route stripped, landmark/goal retained (hand-edited, e.g. "Look for the fitness room. Stop next to the treadmill.") | can it reach the goal without the route? |

### Two prompt-scaffold pipelines (the second variable: is failure the model or the prompt?)
The "scaffold" is the fixed framing text wrapping the instruction in the model prompt. The action-menu wording ("turning left or right by a specific degree, moving forward a certain distance") is **kept byte-identical across both** so the regex action parser and 25 cm / 15° discretization are unaffected. Only the framing and the stop clause differ.

| Pipeline | Scaffold | Stop clause |
|---|---|---|
| **`phase1_baseline`** | unchanged NaVILA framing: *"…robot programmed for **navigation tasks**…"* | "stop if the task is completed" |
| **`phase1_explore`** | exploration-nudged: *"…robot programmed for **search and navigation tasks**… you must actively explore… head toward rooms where the target is likely… avoid revisiting…"* | "stop only once the target is clearly visible in front of you" |

### The 2×2
Each episode runs in all four cells: `{baseline, explore} × {original, goal_only}`.

- `baseline/original` — control; reproduces phase-0 (the only cell that duplicates earlier data)
- `baseline/goal_only` — **key cell:** does it search with route stripped, using its original prompt?
- `explore/original` — does the search scaffold *hurt* normal route-following?
- `explore/goal_only` — does the exploration nudge *rescue* search?

The contrast tells us whether any failure to search is **the model** (fails in both scaffolds) or **just the prompt** (fails baseline, works explore → trivially fixable).

---

## 3. Episode sample

**17 episodes** (R2R-CE `val_unseen`, seed 42, from the phase-0 success pool — episodes NaVILA solved under the original instruction):

`53, 60, 63, 190, 200, 288, 431, 464, 507, 777, 1088, 1187, 1254, 1378, 1388, 1389, 1683`

Dropped from the phase-0 set of 20: **218, 240, 1011** (240 was left unconverted; 218/1011 removed as not relevant).

Restricting to success-pool episodes makes `baseline/original` a true 100%-SR control by construction, so any drop in `goal_only` isolates the route as the cause.

Config: [phase1_instructions.json](phase1_instructions.json) — each entry has `episode_id`, `original`, `goal_only`.

---

## 4. What is held identical to the standard NaVILA eval

So Phase-1 behavior is directly comparable to phase-0 / the paper:
- Model, checkpoint, decoding (greedy, `temperature=0.0`, `max_new_tokens=32`)
- 8-frame video observation, image preprocessing
- Regex action parsing, 25 cm / 15° discretization
- `MAX_EPISODE_STEPS = 500` (default; route-following successes use ~70 steps, so ~7× headroom for search). Raise and re-run `goal_only` only if step-cap timeouts dominate.
- Goal location / success measurement come from the **episode**, not the instruction text — so editing the instruction does not affect what counts as success.

The **only** things that vary: (a) instruction text (`original` vs `goal_only`), (b) scaffold framing + stop clause (`baseline` vs `explore`).

---

## 5. What is recorded

**Per rollout** → [phase1_results.json](phase1_results.json) (flushed per-episode, resume-safe):
- `trajectory` — every low-level step: `action` + agent `[x,y,z]` position (enables coverage/exploration metrics offline)
- `all_action_outputs` — every raw model decision string
- `used_instruction`, `success`, `spl`, `distance_to_goal`, `oracle_success`, `trajectory_length`, `num_steps`
- `action_histogram` — forward/left/right/stop counts (degenerate-behavior signal: all-turns = spinning, single-stop = immediate quit)
- `start_position`, `final_position`, plus `pipeline`/`condition`/`episode_id` tags

**Per episode** → `phase1_videos/<pipeline>/<condition>/episode=<id>-…-spl=<x>.mp4`
- Egocentric view + top-down map (path drawn) + instruction overlaid on every frame (self-labeling)

### Metrics to derive (search-specific; SR alone will mislead)
- SR / SPL / NE per cell (the `original`→`goal_only` drop = route-dependence)
- Exploration: area/navmesh coverage, distinct rooms entered, visit-entropy (from `trajectory` positions)
- Degenerate-behavior rate: % immediate-stop, % beeline-then-stop, % spin/oscillate (from `action_histogram`)
- Approach-conditioned SR: among episodes where the target enters view, does it then reach + stop?
- Stop precision: when it STOPs, was the target actually in range?

---

## 6. How outcomes map to the decision

- **Sensible search** (coverage rises, approaches visible targets, grounded stops, SR>0 on `goal_only`) → keep NaVILA, fine-tune onto search-with-budgets. Cheapest path, no model switch.
- **Approach-only** (works when target visible, degenerate when not — the predicted result) → keep NaVILA, *add* exploration via fine-tuning or a frontier/explorer module.
- **Degenerate** (immediate-stop / beeline / spin regardless of FOV, in *both* scaffolds) → no transfer; consider an ObjectNav policy or heavier retraining / model switch.

This is the prerequisite for the broader thread (cf. phase 0's **time-budget language**): the endgame is **budgeted search** ("find X within N seconds/steps"). Phase 1 answers *does it search at all* before *does it search within budget*.

---

## 7. Caveats

- **8-frame memory ceiling** — caps how long a coherent search can run, regardless of prompt/fine-tuning.
- **Instruction phrasing sensitivity** — it's a VLM; `goal_only` wording was hand-authored per episode and is not phrasing-controlled. A null result under one phrasing is not conclusive.
- **Scene data is read from the external HDD** via a symlink (`data/scene_datasets/mp3d` → `/media/maitree-tiamat/Expansion/NaVILA_data/scene_datasets/mp3d`); the HDD must stay mounted for the run.
- **Step cap** — 500 may still under-serve genuine search; treat timeout-heavy `goal_only` failures as a reason to raise the cap and re-run, not as conclusive.
- `baseline/original` duplicates phase-0 data (kept as an in-run sanity control).

---

## 8. How to run

```bash
cd ~/NaVILA/evaluation
conda activate navila
python run_phase1.py \
  --pipelines phase1_baseline,phase1_explore \
  --conditions original,goal_only \
  --gpu 0 \
  --output phase1_results.json \
  --video-dir phase1_videos
# resume-safe; re-running skips completed (pipeline, condition, episode) cells.
# --max-steps N   to raise the step cap for search
# --max-episodes N / --pipelines / --conditions   to run a subset
```

Runner: [run_phase1.py](run_phase1.py).

---

## 9. Results (run 2026-06-11)

All 68 rollouts completed (17 episodes × 2 pipelines × 2 conditions). Results in
[phase1_results.json](phase1_results.json); analysis via [analyze_phase1.py](analyze_phase1.py).

| Cell | SR | SPL | NE (m) | Avg steps | Coverage (m²) | Wander | Revisit | Degenerate |
|---|---|---|---|---|---|---|---|---|
| `baseline/original` *(control)* | **100%** | 0.95 | 1.42 | 50 | 4.6 | 1.31 | 2.84 | — |
| `baseline/goal_only` | **35%** | 0.33 | 6.27 | 147 | 3.9 | 1.65 | 12.4 | 3 cap, 2 spin |
| `explore/original` | **82%** | 0.77 | 2.10 | 61 | 4.8 | 1.46 | 3.20 | — |
| `explore/goal_only` | **35%** | 0.33 | 8.04 | 255 | 5.0 | 1.39 | 47.8 | 6 cap |

- *Coverage* = distinct 0.5 m (x,z) cells visited × 0.25 m². *Wander* = path_length / net displacement. *Revisit* = steps / distinct cells (higher = more churning over the same ground).

### Findings

1. **Control reproduces phase-0 (100% SR, SPL 0.95).** Pipeline is sound; the rest is comparable.

2. **NaVILA does goal-only search at ~35% zero-shot — not degenerate.** With the route stripped it still reaches ~6/17 targets, tripling its steps (50→147). Only ~5/17 show degenerate flags. It is *trying*, not freezing.

3. **But the extra motion is revisiting, not exploring.** Coverage stays flat (~4–5 m²) across every cell regardless of step count — the goal-only/explore cells take 3–5× more steps yet cover the *same area* as the 50-step control. Revisit ratio climbs 2.8 → 12.4 → **47.8**. With no map and only 8 frames of memory, the agent churns over seen ground instead of systematically uncovering new space. This is the missing skill: **productive exploration**, not motion.

4. **The exploration scaffold is not the fix — failure is the model, not the prompt.** The nudged scaffold left goal-only success unchanged (**35% → 35%**), made wandering far worse (revisit 12.4 → 47.8, NE 6.27 → 8.04, more step-cap timeouts), and *hurt* normal route-following (**100% → 82%**). Because failure persists in both scaffolds, prompt engineering cannot recover search.

### Per-episode pattern (`baseline/goal_only`)

- **Successes are mostly short approaches** (28–78 steps; target near/visible → walk to it): ep 53, 63, 464, 1088, 1388. One genuine *longer* search succeeded — ep **1683** (188 steps, NE 0.3).
- **Failures split three ways:** wandered to the 500-step cap without recognizing/stopping (190, 288, 1187 — e.g. *"look for the pool table"* explored 500 steps, never terminated); spun in place (1254, 1378); or quit early far from goal (60, 200, 431, 507, 777, 1389; NE 4–20 m).
- Read: **near target → approaches and succeeds; needs real exploration → wanders/spins/quits.** Confirms the approach-vs-explore split.

Telling videos (`phase1_videos/phase1_baseline/goal_only/`): `episode=1683` (productive search success), `episode=1088` (clean approach), `episode=288` (500-step non-terminating wander), `episode=1254` (spin). Compare against `phase1_baseline/original/` for the same ids to see the route's contribution.

## 10. Conclusion → decision

NaVILA lands in the predicted **"approach-capable, weak-explorer"** regime: it can reach goals that are near/visible (~35% zero-shot) and is not degenerate, but it lacks *productive exploration* — it revisits rather than uncovers, and recognizes-and-stops unreliably. Crucially, this is a **model limitation, not a prompt limitation** (the exploration scaffold did not help and hurt route-following).

**Decision: keep NaVILA; do not switch models.** Exploration must be *added* — via fine-tuning toward search (the budgeted-search direction) and/or pairing with a frontier/explorer + memory module — rather than prompted in. This validates moving to the search-with-budgets fine-tuning track, with these 17 episodes as a before/after probe.

To reproduce the analysis:
```bash
cd ~/NaVILA/evaluation && conda activate navila
python analyze_phase1.py --input phase1_results.json
```
