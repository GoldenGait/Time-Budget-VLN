# Data schema: budget-conditioned multi-goal VLN

Two layers. Layer 1 is our own rich metadata (source of truth, easy to debug/split).
Layer 2 is the exact format NaVILA's trainer eats, derived from Layer 1 by the flattener.

## Layer 1 — episode metadata (our format)

One record per sampled episode. This is what `generate_episodes.py` produces (plus a
few fields the flattener adds). Geometry + costs only; no frames, no language.

```jsonc
{
  "episode_id": "GdvgFV5R1Z5_ep0",
  "scene": "GdvgFV5R1Z5",
  "split": "train",                  // train | val | test_heldout — for the signature eval
  "seed": 0,
  "start_pose": { "position": [x,y,z], "yaw": <rad> },
  "goals": [
    { "slot": "G1", "category": "sink",   "center": [x,y,z], "navpoint": [x,y,z], "obj_id": 16 },
    { "slot": "G2", "category": "toilet", "center": [x,y,z], "navpoint": [x,y,z], "obj_id": 86 }
  ],
  "costs": {
    "S_G1": 22, "S_G2": 35,          // fresh-heading single-leg costs (primitive steps)
    "nearest_only": 22,              // = min(S_G1, S_G2)
    "both_tour": 78,                 // continuous 2-leg tour, cheaper ordering
    "ordering_for_both": "G1,G2"
  },
  "action_unit": { "forward_m": 0.25, "turn_deg": 15.0, "success_distance": 0.2 }
}
```

Budget unit = primitive steps (0.25 m forward / 15° turn). The flip boundary for this
episode is the half-open interval `[nearest_only, both_tour)`: any budget in `[22, 78)`
should produce nearest-only behavior; any budget `>= 78` should visit both.

## Layer 2 — NaVILA training annotations (their format)

Verified against the released `R2R/annotations.json` and `llava/data/dataset.py`.
A flat list, **one record per timestep**:

```jsonc
{
  "video_id": "GdvgFV5R1Z5_ep0_tight-0",   // <episode>_<regime>-<step>
  "q": "<budget-conditioned instruction>",  // RAW instruction only; loader wraps it
  "a": "The next action is move forward 75 cm.",
  "frames": ["GdvgFV5R1Z5_ep0_tight/frame_0.jpg", "...frame_1.jpg", ...]  // history to here
}
```

- An N-action episode → N records, with `frames` growing one entry per step.
- `q` holds ONLY the instruction text. At load time NaVILA wraps it with a fixed preamble
  ("Imagine you are a robot... Your assigned task is: \"{q}\" ... decide your next action,
  which could be turning left or right by a specific degree, moving forward a certain
  distance, or stop"). **So the budget text goes inside `q`.**
- `a` vocabulary (fixed by that preamble): `move forward 75 cm` · `turn left N degrees` ·
  `turn right N degrees` · `stop`. 75 cm = 3 primitive forwards; turns are 15/30/45°
  (1/2/3 primitives). The flattener aggregates the expert's primitive stream into these
  native chunks (cap 75 cm forward / 45° turn per action).

### Budget in the instruction (decided)

Live **remaining** budget, re-injected every step, depleting in primitive-step units
(a 75 cm forward costs 3; a turn costs its primitive count):

```
Find the sink, then the toilet. You have 30 steps of budget.          # step 0
Find the sink, then the toilet. You have 27 steps of budget left.     # after 1 forward
...
```

### How one episode becomes the flip

Each episode is flattened **twice**, once per regime, with a budget sampled on each side
of `[nearest_only, both_tour)`:

| regime | sampled budget | target behavior | expert path |
|---|---|---|---|
| `tight` | in `[nearest_only, both_tour)` e.g. 30 | reach nearest goal, then stop | S → nearer goal → stop |
| `loose` | `>= both_tour` e.g. 85 | visit both goals | S → first → second → stop |

Same scene, same goal pair, only the budget number and the action trace differ — that is
the behavior flip, taught directly as supervised data.

## Files / locations

- Layer 1 JSON: `multigoal_episodes/data/episodes_<scene>.json` (small, in repo).
- Layer 2 annotations + frames: written to **HDD** (root disk is near full), under the
  NaVILA-Dataset layout so `datasets_mixture.py` can point `data_path`/`image_path` at it.
