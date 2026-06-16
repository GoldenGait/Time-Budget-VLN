# Phase 0 Diagnostic — Time-Budget Language in NaVILA

**Goal:** Test, zero-shot, whether NaVILA responds at all to time-budget language appended to existing R2R val_unseen instructions. No fine-tuning, no model changes.

**Date:** 2026-05-28
**Researcher:** Maitree
**Model:** `navila-llama3-8b-8f` (LLaMA-3 8B + SigLIP, 8 video frames) — checkpoint at `~/models/navila-llama3-8b-8f`
**Split:** R2R-CE val_unseen
**Sample:** 20 episodes (seed=42), filtered to episodes NaVILA succeeded in under the original instruction (success pool: 1002 / 1839)

---

## 1. Experimental design

Three conditions per episode (each is an independent Habitat-Sim rollout, same model, same simulator config, same decoding parameters):

| Condition | Instruction modification |
|---|---|
| **A — Original** | unmodified `val_unseen` instruction |
| **B — Short budget** | original + `" You have 15 seconds."` |
| **C — Long budget** | original + `" You have 3 minutes."` |

The 15-second budget is **deliberately infeasible** for most episodes; the 3-minute budget is effectively unlimited. The two extremes maximize the chance of detecting any behavioral response.

The only thing that varies between conditions is the instruction text fed to the model. The full prompt scaffold, image preprocessing, decoding parameters (greedy, max_new_tokens=32), regex action parsing, and 25-cm / 15-degree action discretization are byte-identical to the standard NaVILA eval.

### Why success-filter the sample

If the model already fails on an episode under the original instruction, the budget conditions cannot tell us anything new. Filtering to success-only episodes makes the "original" column a true control (100% SR by construction) and isolates the budget phrase as the only candidate cause of any change.

### Baseline reproduction

Before Phase 0, the full val_unseen baseline (1839 episodes) was run to confirm the pipeline matches the paper:

| Metric | Paper (Table I) | Repro | Δ |
|---|---|---|---|
| NE (m) | 5.22 | 5.23 | +0.01 |
| OS | 62.5% | 61.56% | −0.94 |
| **SR** | **54.0%** | **54.49%** | **+0.49** |
| **SPL** | **49.0%** | **49.89%** | **+0.89** |

Match within 1pp on all reported metrics — pipeline is faithful.

---

## 2. Sampled episodes

20 success-pool episodes (sorted ascending): `53, 60, 63, 190, 200, 218, 240, 288, 431, 464, 507, 777, 1011, 1088, 1187, 1254, 1378, 1388, 1389, 1683`

Full preview of original vs. modified instructions: [phase0_instructions_preview.json](phase0_instructions_preview.json)

---

## 3. Aggregate results

### 3.1 Summary metrics

| Condition | SR | SPL | Avg steps (low-level) | Avg trajectory length |
|---|---|---|---|---|
| Original | **100.0%** | 0.953 | 50.0 | 8.67 m |
| Short budget | **85.0%** | 0.823 | **114.2** | 8.20 m |
| Long budget | **90.0%** | 0.870 | 93.9 | 8.77 m |

### 3.2 Pass-rate comparison

| Condition | Passed | Pass % | Δ vs Original | Failed episodes |
|---|---|---|---|---|
| Original | 20 / 20 | 100% | — | — |
| Short budget | 17 / 20 | **85%** | **−15pp** | 777, 1011, 1254 |
| Long budget | 18 / 20 | **90%** | **−10pp** | 777, 1011 |

All three failures hit the 500-step episode cap — the model wandered until it ran out of simulator-allotted steps.

### 3.3 Action distribution (low-level steps, summed across the 20 episodes)

| Condition | forward | turn_left | turn_right | stop | Σ steps |
|---|---|---|---|---|---|
| Original | 685 (68.6%) | 150 (15.0%) | 144 (14.4%) | 20 (2.0%) | 999 |
| Short budget | 1130 (49.5%) | **577 (25.3%)** | **560 (24.5%)** | 17 (0.7%) | 2284 |
| Long budget | 1160 (61.8%) | 346 (18.4%) | 354 (18.8%) | 18 (1.0%) | 1878 |

Under the **short** budget, turn-left + turn-right combined goes from **29% → 50%** of all low-level actions. The model literally rotates twice as much. Under the **long** budget the shift is milder but in the same direction.

---

## 4. Per-episode trajectory length and steps

Trajectory length (m) and low-level step counts under each condition. `⚠️` marks an episode where success flipped from the original.

| Ep | Traj-A | Traj-B | Traj-C | Steps-A | Steps-B | Steps-C | Success A/B/C |
|---|---|---|---|---|---|---|---|
| 53 | 12.54 | 12.79 | 12.79 | 71 | 71 | 71 | ✓ / ✓ / ✓ |
| 60 | 8.27 | 7.52 | 7.02 | 46 | 42 | 40 | ✓ / ✓ / ✓ |
| 63 | 7.25 | 7.25 | 7.25 | 44 | 44 | 44 | ✓ / ✓ / ✓ |
| 190 | 12.03 | 10.53 | 10.53 | 60 | 63 | 63 | ✓ / ✓ / ✓ |
| 200 | 4.66 | 4.96 | 4.96 | 54 | 53 | 51 | ✓ / ✓ / ✓ |
| 218 | 6.71 | 7.46 | 7.46 | 40 | 39 | 39 | ✓ / ✓ / ✓ |
| 240 | 13.50 | 14.25 | 15.00 | 62 | 66 | 69 | ✓ / ✓ / ✓ |
| 288 | 9.25 | 10.21 | 10.21 | 45 | 53 | 53 | ✓ / ✓ / ✓ |
| 431 | 12.27 | 11.77 | 12.16 | 69 | 66 | 71 | ✓ / ✓ / ✓ |
| 464 | 4.67 | 4.78 | 4.73 | 36 | 36 | 37 | ✓ / ✓ / ✓ |
| 507 | 6.50 | 7.22 | 7.22 | 42 | 45 | 45 | ✓ / ✓ / ✓ |
| **777** | 11.98 | **19.87** | **20.13** | 65 | **500** | **500** | ✓ / ✗ / ✗ ⚠️ |
| **1011** | 8.16 | 4.38 | 4.38 | 47 | **500** | **500** | ✓ / ✗ / ✗ ⚠️ |
| 1088 | 5.25 | 5.25 | 5.25 | 28 | 28 | 28 | ✓ / ✓ / ✓ |
| 1187 | 8.47 | 8.47 | 8.47 | 36 | 36 | 36 | ✓ / ✓ / ✓ |
| **1254** | 12.87 | 3.15 | 12.90 | 75 | **500** | 85 | ✓ / ✗ / ✓ ⚠️ |
| 1378 | 4.51 | 5.26 | 5.26 | 28 | 30 | 30 | ✓ / ✓ / ✓ |
| 1388 | 7.75 | 4.50 | 5.25 | 54 | 32 | 36 | ✓ / ✓ / ✓ |
| 1389 | 7.75 | 5.50 | 5.50 | 54 | 37 | 37 | ✓ / ✓ / ✓ |
| 1683 | 8.97 | 8.97 | 8.97 | 43 | 43 | 43 | ✓ / ✓ / ✓ |

**Identical trajectories across all 3 conditions:** 5 / 20 (53, 63, 1088, 1187, 1683). For these the model produced byte-identical action sequences regardless of the appended budget phrase.

---

## 5. Divergent episodes (Δtraj > 1 m or success flipped)

8 / 20 episodes diverged.

### Robust failures under any budget — 777, 1011
Both budget phrases cause the model to wander until the 500-step cap, despite easily solving these under the original instruction. Watch:
- [phase0_videos/original/episode=777-*.mp4](phase0_videos/original/) vs `short_budget/` and `long_budget/`
- Same for episode 1011

### Budget-sensitive — 1254
Fails only under "15 seconds" (3.15 m / 500 steps), succeeds under both "original" (12.87 m / 75 steps) and "3 minutes" (12.90 m / 85 steps). The *kind* of budget matters here, not just the presence of one.

### Shorter under budget — 60, 190, 1388, 1389
Trajectory length **decreases** when the budget is appended; the model finds a tighter path. Of these, 1388/1389 see the largest reductions (~3 m / ~20 steps each) and SPL actually *improves* to 1.0.

### Longer under budget — 240
Trajectory grows from 13.5 m (original) to 14.25 (short) to 15.0 m (long). SPL drops accordingly. Same task, monotonically more wandering as the budget phrase changes.

---

## 6. Interpretation

1. **The budget phrase reaches the model's behavior.** Not just the action histogram (which shifts dramatically under "15 seconds": 68.6%→49.5% forward, 29%→50% turning) but the success rate itself (−15pp under short, −10pp under long).

2. **NaVILA does not interpret "15 seconds" as a hurry signal.** If it did, we'd expect *fewer* steps and earlier `stop` calls. We see the opposite — average step count **more than doubles** under the short budget. The phrase appears to make the model less confident in forward progress.

3. **The effect is partly destructive, partly creative.** 3 episodes fail under the short budget (wandering to the 500-step cap), but 4 episodes get *shorter* trajectories — for some episodes the budget seems to function as a discount on extraneous exploration.

4. **Long budget is intermediate.** The phrase still degrades behavior (−10pp SR, +88% steps) but less than the short budget. There's a graded response, not a binary one.

This is a **positive signal for Phase 1**: budget-phrase-conditional fine-tuning has a real lever to pull. The model can clearly read "you have X seconds" but lacks a learned mapping from those tokens to coherent time-pressured behavior.

---

## 7. Files

- **[run_phase0.py](run_phase0.py)** — runs all 3 conditions; supports `--success-from`, `--video-dir`, deterministic seeding, incremental save (each episode flushed to disk).
- **[analyze_phase0.py](analyze_phase0.py)** — produces the summary table, per-episode trajectory comparison, action histograms, divergent-episode listing.
- **[phase0_results.json](../../phase0_results.json)** — 60 records (3 conditions × 20 eps). Each has: `original_instruction`, `modified_instruction`, `all_action_outputs` (per high-level model decision), `action_histogram`, `num_steps`, `trajectory_length`, `success`, `spl`, `distance_to_goal`, `start_position`, `final_position`, `trajectory` (per low-level step: action + [x,y,z] position).
- **[phase0_instructions_preview.json](phase0_instructions_preview.json)** — all 20 episodes × 3 conditions, instruction text only.
- **[phase0_videos/{original,short_budget,long_budget}/<ep_id>.mp4](phase0_videos/)** — 60 MP4s for human inspection. Each frame shows RGB observation + top-down map + the modified instruction overlaid.
- **[phase0.log](../../phase0.log)** — full run log.

### Reproduce

```bash
cd ~/NaVILA/evaluation
CUDA_VISIBLE_DEVICES=0 /home/maitree-tiamat/miniconda3/envs/navila/bin/python run_phase0.py \
    --gpu 0 --seed 42 --num-episodes 20 --split val_unseen \
    --success-from /home/maitree-tiamat/NaVILA/evaluation/eval_out/navila-llama3-8b-8f/VLN-CE-v1/val_unseen \
    --video-dir /home/maitree-tiamat/NaVILA/evaluation/phase0_videos \
    --output /home/maitree-tiamat/phase0_results.json
/home/maitree-tiamat/miniconda3/envs/navila/bin/python analyze_phase0.py \
    --input /home/maitree-tiamat/phase0_results.json
```

Full rollout time on a free RTX 5090: ~25 minutes.
