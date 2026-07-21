"""Orchestrate Layer 1 + Layer 2 across many MP3D scenes.

For each scene (until --target viable scenes are collected):
  1. Layer 1: generate episodes (reuses an existing data/episodes_<scene>.json
     if present, so the pinned starter scene is preserved).
  2. keep the scene only if it yields >= --min-episodes episodes.
  3. Layer 2: flatten that scene's episodes into the NaVILA SFT dataset on HDD.

Held-out test scenes are NOT produced here: they are drawn later from the
remaining untouched scenes, so train/test scene sets are disjoint by
construction.

Run under the navila conda env, from the repo root:
  python multigoal_episodes/scripts/build_dataset.py --target 20
"""
import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = "multigoal_episodes/data"
MP3D = "/media/maitree-tiamat/Expansion/NaVILA_data/scene_datasets/mp3d"
STARTER = "GdvgFV5R1Z5"


def scene_list():
    """Starter first, then the rest of the 90 MP3D scenes in sorted order."""
    all_scenes = sorted(os.listdir(MP3D))
    ordered = [STARTER] + [s for s in all_scenes if s != STARTER]
    return ordered


def ep_count(path):
    try:
        return len(json.load(open(path))["episodes"])
    except Exception:
        return 0


def run(cmd):
    print("  $", " ".join(cmd), flush=True)
    return subprocess.run(cmd).returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=20, help="how many viable scenes")
    ap.add_argument("--n", type=int, default=20, help="episodes per scene")
    ap.add_argument("--min-episodes", type=int, default=12,
                    help="keep a scene only if it yields at least this many")
    ap.add_argument("--max-tries", type=int, default=600)
    ap.add_argument("--scenes", nargs="*", default=None,
                    help="explicit scene ids (default: auto from MP3D dir)")
    args = ap.parse_args()

    os.makedirs(DATA, exist_ok=True)
    candidates = args.scenes or scene_list()
    py = sys.executable
    gen = os.path.join(HERE, "generate_episodes.py")
    flat = os.path.join(HERE, "build_sft_dataset.py")

    collected, skipped = [], []
    for scene in candidates:
        if len(collected) >= args.target:
            break
        out = f"{DATA}/episodes_{scene}.json"

        # Layer 1 (reuse existing JSON, e.g. the pinned starter)
        if os.path.exists(out) and ep_count(out) >= args.min_episodes:
            print(f"[{scene}] reuse existing ({ep_count(out)} eps)", flush=True)
        else:
            print(f"[{scene}] Layer 1 ...", flush=True)
            run([py, gen, scene, "--n", str(args.n), "--split", "train",
                 "--max-tries", str(args.max_tries), "--out", out])

        c = ep_count(out)
        if c < args.min_episodes:
            print(f"[{scene}] SKIP (only {c} eps)", flush=True)
            skipped.append((scene, c))
            if os.path.exists(out) and scene != STARTER:
                os.remove(out)
            continue

        # Layer 2
        print(f"[{scene}] Layer 2 ({c} eps) ...", flush=True)
        run([py, flat, scene, "--json", out])
        collected.append((scene, c))
        print(f"=== collected {len(collected)}/{args.target}: {scene} ({c} eps) ===\n",
              flush=True)

    print("\n================ SUMMARY ================")
    print(f"collected {len(collected)} scenes:")
    for s, c in collected:
        print(f"  {s}: {c} episodes")
    if skipped:
        print(f"skipped {len(skipped)}: " + ", ".join(f"{s}({c})" for s, c in skipped))


if __name__ == "__main__":
    main()
