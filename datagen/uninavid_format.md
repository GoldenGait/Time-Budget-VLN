# uninavid_format.md — Phase 0 ground truth (extracted from code + released data, not assumed)

Sources of truth:
- Released finetune sample: `Uni-NaVid/model_zoo/_navf/Nav-Finetune/open_uninavid_sampled_500.json` (500 records, all `NAV_ID`).
- Eval harness (reconstructed ObjectNav loop): `objectnav_eval/objectnav_uninavid.py`.

---

## 1. Record schema

```json
{
  "id":    "NAV_ID_VLN_35150_002",
  "video": "nav_videos/35150_002.mp4",
  "conversations": [
    {"from": "human", "value": "Imagine you are a robot ... <image> ... Your assigned task is: '<INSTRUCTION>'. Analyze this series of images to determine your next four actions. The predicted action should be one of the following: forward, left, right, or stop."},
    {"from": "gpt",   "value": "right right forward forward"}
  ]
}
```
- `id`: string, contains `NAV_ID` (the loader keys nav-specific augmentation off this).
- `video`: relative path to an encoded **`.mp4`** in `nav_videos/`. (Not a frames dir.)
- `human.value`: the fixed **PROMPT_WRAP** with a single `<image>` token and the task string embedded in `'...'`.
- `gpt.value`: **space-separated action tokens**, vocabulary `{forward, left, right, stop}`.

Verbatim gpt answers (nav records): `"right right forward forward"`, `"forward forward right forward"`, `"left left forward right"`, `"right forward right forward"`.

**Action-label granularity: exactly 4 tokens per answer — confirmed across all 500 records (100%).**
There is NO NaVILA-style phrasing ("move forward 25 cm / turn left 30 degrees"); Uni-NaVid uses bare tokens.

Action semantics (from `objectnav_uninavid.make_sim`): `forward` = move_forward **0.25 m**; `left`/`right` = turn **30°**; `stop` = end episode.

## 2. Video format
- Encoded `.mp4` under `nav_videos/`; Uni-NaVid's loader samples at **fps = 1** and resizes to **224** via the `clip-patch14-224` image processor.
- Source mp4 resolution is NOT binding (loader resizes) — released videos not on disk to measure. **Generate at the eval sensor resolution (below) and encode to `.mp4`.**

---

## 3. Cadence spec (THE Phase-0 addendum — flattener record grid MUST mirror this)

From `objectnav_uninavid.py` control loop (L177-195):
- The model is prompted for **4** actions ("your next four actions") and the gpt label carries **4**.
- **The harness executes only the FIRST 2, then re-infers** (`pending` is capped at 2: `if len(pending) == 2: break`; re-`predict()` only `if not pending`).
- A frame is captured **before every primitive** (`add_frame` at loop top). Each `predict()` consumes the ~2 new frames observed since the last inference; the online token-merge (`feat_cache`) holds the full history internally. The very first inference sees 1 frame.
- **STOP is honored the moment it is popped** (chunk position 1 or 2) and ends the episode immediately.
- Forced episode end at the Habitat cap (eval `max_steps = 500`).

### Consequences (locked)
1. **Inference / record boundary = every 2 executed primitives.**
2. **4-action labels overlap and slide by 2:** record at primitive `t` labels prims `t..t+3`; the next record at `t+2` labels `t+2..t+5`.
3. **Budget accounting — corrects the earlier "4 emitted consume 4" convention:** only **2** primitives are consumed per inference. So
   - `N_t = B − t`, where `t` = executed primitives so far,
   - `t` advances by **2** per record ⟹ `N` decrements by **2 per record**,
   - the instruction restates the current `N` at **each inference** (every 2 primitives).
4. **RECORD GRID for the flattener:** emit one record per inference (every 2 primitives); `gpt` label = the next **4** expert primitives, padded with `stop` past trace end; `human` instruction carries `N_t`. Force a record boundary exactly at the truncation step in **both** branches (tight/loose) so the decision-point pair exists.
5. **Decision-point parity:** because eval only decides every 2 primitives, the truncation/decision step must land on an **even primitive index** (an inference boundary), or the trained stop can't be reproduced at eval. The turn-to-center turns (≤3) count as primitives and shift parity — land the final `stop` on an even index.

---

## 4. Sensor spec — SINGLE SOURCE OF TRUTH for generation (from `objectnav_uninavid.make_sim`)

| param | value |
|---|---|
| sensor | RGB, `PINHOLE` |
| resolution | **640 × 480** (W×H) |
| HFOV | **79°** |
| camera height | **0.88 m** |
| orientation | forward (0,0,0) |
| move_forward | **0.25 m** |
| turn_left / turn_right | **30°** |
| navmesh | `agent_radius 0.18`, `agent_height 0.88` |
| model input | 224 (image_processor resizes; render 640×480 → processor → 224, same path as eval) |

**Detection "facing" cone = forward half-HFOV = 79/2 = 39.5°** (NOT 45°). Use 39.5° for the facing check. Full detection (locked): any same-category instance within **geodesic < 1.0 m** AND center within **±39.5°** of forward AND clear line-of-sight AND **≥ 50** target semantic pixels; then turn-to-center (≤3 turn primitives, counted in `T`) and `stop`.

---

## 5. Instruction template (locked, Phase 4)
`"Search for a {article} {object}. You have {N} steps remaining."` — article + object surface form exactly as the released records render them; `N` a bare integer. **Byte-equality unit test (Gate P4):** the string rendered for a given state by the training flattener and by the eval harness must be identical.

## 6. Open follow-ups (not blocking cadence)
- MP3D generation needs a working **semantic sensor** (target-pixel detection + non-target object voting). We have MP3D `house_segmentations` (regions/objects) + render `.glb`, but the Habitat semantic-sensor path for MP3D must be verified before MP3D search generation (HM3D is clean via `.semantic.glb`).
