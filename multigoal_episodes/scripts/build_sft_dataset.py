"""Layer 2 — flatten budget-conditioned episodes into NaVILA SFT annotations.

Reads a Layer-1 episodes_<scene>.json (with stored expert primitive traces) and,
for each episode x regime (tight / loose), replays the trace to:
  1. dump one 512x512 RGB frame per native-action decision point,
  2. aggregate 0.25m/15deg primitives into NaVILA's native vocabulary
     (move forward 25/50/75 cm, turn left/right 15/30/45 degrees, stop),
  3. write the budget-conditioned instruction with the LIVE remaining budget,
  4. emit one {video_id, q, a, frames} record per action (+ a final stop).

Output mirrors the NaVILA-Dataset layout so datasets_mixture.py can point at it:
  <out>/annotations.json          (data_path)
  <out>/frames/<video>/frame_K.jpg (image_path = <out>/frames)

Usage:
  python build_sft_dataset.py GdvgFV5R1Z5 --episode 0      # just ep0, to eyeball
  python build_sft_dataset.py GdvgFV5R1Z5                  # all episodes
"""
import argparse
import json
import math
import os

import numpy as np
import habitat_sim
from habitat_sim.utils.common import quat_from_angle_axis
from PIL import Image

FORWARD_M, TURN_DEG = 0.25, 15.0
FRAME_RES = 512                       # match released R2R frames (512x512 RGB)
MAX_CHUNK = 3                         # 3 primitives = 75 cm forward / 45 deg turn
DEFAULT_OUT = "/media/maitree-tiamat/Expansion/NaVILA_data/budget_vln_sft"


# ---------------------------------------------------------------- sim (rgb only)
def make_sim(scene_id, mp3d):
    glb = f"{mp3d}/{scene_id}/{scene_id}.glb"
    backend = habitat_sim.SimulatorConfiguration()
    backend.scene_id = glb
    backend.enable_physics = False
    s = habitat_sim.CameraSensorSpec()
    s.uuid, s.sensor_type = "rgb", habitat_sim.SensorType.COLOR
    s.resolution = [FRAME_RES, FRAME_RES]
    s.position = [0.0, 1.25, 0.0]
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [s]
    agent_cfg.action_space = {
        "move_forward": habitat_sim.agent.ActionSpec(
            "move_forward", habitat_sim.agent.ActuationSpec(amount=FORWARD_M)),
        "turn_left": habitat_sim.agent.ActionSpec(
            "turn_left", habitat_sim.agent.ActuationSpec(amount=TURN_DEG)),
        "turn_right": habitat_sim.agent.ActionSpec(
            "turn_right", habitat_sim.agent.ActuationSpec(amount=TURN_DEG)),
    }
    return habitat_sim.Simulator(habitat_sim.Configuration(backend, [agent_cfg]))


def set_agent(sim, pos, yaw):
    st = habitat_sim.AgentState()
    st.position = np.asarray(pos, dtype=np.float32)
    st.rotation = quat_from_angle_axis(yaw, np.array([0.0, 1.0, 0.0]))
    sim.get_agent(0).set_state(st)


def grab_rgb(sim):
    return sim.get_sensor_observations()["rgb"][:, :, :3].copy()


# ---------------------------------------------------------------- aggregation
def aggregate(prims):
    """0.25m/15deg primitive stream -> NaVILA native actions.
    Returns list of {"phrase", "prims"} where prims is the primitive cost."""
    out, i, n = [], 0, len(prims)
    while i < n:
        a = prims[i]
        j = i
        while j < n and prims[j] == a:
            j += 1
        run = j - i
        while run > 0:
            k = min(MAX_CHUNK, run)
            if a == "move_forward":
                phrase = f"move forward {k * 25} cm"
            elif a == "turn_left":
                phrase = f"turn left {k * 15} degrees"
            elif a == "turn_right":
                phrase = f"turn right {k * 15} degrees"
            else:
                raise ValueError(f"unknown primitive {a!r}")
            out.append({"phrase": phrase, "prims": k})
            run -= k
        i = j
    return out


# ---------------------------------------------------------------- budgets / text
def sample_budgets(nearest_only, both_tour, tight_frac, loose_margin):
    """tight in [nearest_only, both_tour); loose >= both_tour, both as int steps."""
    span = both_tour - nearest_only
    tight = nearest_only + math.ceil(span * tight_frac)
    tight = max(nearest_only, min(tight, both_tour - 1))
    loose = math.ceil(both_tour * (1.0 + loose_margin))
    return int(tight), int(loose)


def instruction(c1, c2, remaining, first):
    goals = f"the {c1} and the {c2}"
    if first:
        return f"Find {goals}. You have a budget of {remaining} steps."
    return f"Find {goals}. You have {remaining} steps of budget left."


# ---------------------------------------------------------------- flatten one
def flatten_regime(sim, ep, regime, budget, frames_root, c1, c2, write_frames=True):
    """Replay one regime's stored trace; dump frames; return list of records."""
    trace = ep["traces"][regime]
    prims = trace["actions"]
    native = aggregate(prims)

    video = f"{ep['episode_id']}_{regime}"
    vdir = os.path.join(frames_root, video)
    if write_frames:
        os.makedirs(vdir, exist_ok=True)

    # replay: frame_0 at start, then one frame after each native action
    set_agent(sim, ep["start_pose"]["position"], ep["start_pose"]["yaw"])
    frame_paths = []

    def save(idx):
        fn = f"frame_{idx}.jpg"
        if write_frames:
            Image.fromarray(grab_rgb(sim)).save(os.path.join(vdir, fn), quality=90)
        frame_paths.append(f"{video}/{fn}")

    save(0)
    cursor = 0
    for act in native:
        for _ in range(act["prims"]):
            sim.step(prims[cursor]); cursor += 1
        save(len(frame_paths))

    # build per-decision records (one per native action, then stop)
    records, remaining = [], budget
    for k, act in enumerate(native):
        records.append({
            "video_id": f"{video}-{k}",
            "q": instruction(c1, c2, remaining, first=(k == 0)),
            "a": f"The next action is {act['phrase']}.",
            "frames": frame_paths[:k + 1],
        })
        remaining -= act["prims"]
    records.append({
        "video_id": f"{video}-{len(native)}",
        "q": instruction(c1, c2, remaining, first=(len(native) == 0)),
        "a": "The next action is stop.",
        "frames": frame_paths[:len(native) + 1],
    })
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene")
    ap.add_argument("--episode", type=int, default=None,
                    help="flatten only this episode index (default: all)")
    ap.add_argument("--json", default=None)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--mp3d",
                    default="/media/maitree-tiamat/Expansion/NaVILA_data/scene_datasets/mp3d")
    ap.add_argument("--tight-frac", type=float, default=0.4,
                    help="tight budget = nearest + frac*(both-nearest)")
    ap.add_argument("--loose-margin", type=float, default=0.15,
                    help="loose budget = ceil(both_tour * (1+margin))")
    ap.add_argument("--print-only", action="store_true",
                    help="print records to stdout, do not write frames/annotations")
    args = ap.parse_args()

    jpath = args.json or f"multigoal_episodes/data/episodes_{args.scene}.json"
    data = json.load(open(jpath))
    episodes = data["episodes"]
    if args.episode is not None:
        episodes = [episodes[args.episode]]

    frames_root = os.path.join(args.out, "frames")
    if not args.print_only:
        os.makedirs(frames_root, exist_ok=True)

    sim = make_sim(args.scene, args.mp3d)
    all_records = []
    for ep in episodes:
        c1, c2 = ep["G1"]["category"], ep["G2"]["category"]
        tight_b, loose_b = sample_budgets(
            ep["nearest_only"], ep["both_tour"], args.tight_frac, args.loose_margin)
        budgets = {"tight": tight_b, "loose": loose_b}
        for regime in ("tight", "loose"):
            recs = flatten_regime(sim, ep, regime, budgets[regime],
                                  frames_root, c1, c2,
                                  write_frames=not args.print_only)
            all_records.extend(recs)
            print(f"{ep['episode_id']} {regime}: budget={budgets[regime]} "
                  f"-> {len(recs)} records "
                  f"(nearest={ep['nearest_only']} both={ep['both_tour']})")
    sim.close()

    if args.print_only:
        print(json.dumps(all_records, indent=2))
        return

    ann = os.path.join(args.out, "annotations.json")
    # append-merge if file exists (so per-episode runs accumulate)
    existing = json.load(open(ann)) if os.path.exists(ann) else []
    by_id = {r["video_id"]: r for r in existing}
    for r in all_records:
        by_id[r["video_id"]] = r
    merged = list(by_id.values())
    json.dump(merged, open(ann, "w"), indent=2)
    print(f"\nwrote {len(all_records)} new records "
          f"({len(merged)} total) -> {ann}")
    print(f"frames root (image_path) -> {frames_root}")


if __name__ == "__main__":
    main()
