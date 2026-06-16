#!/usr/bin/env python3
"""Phase 1 analysis: SR/SPL/NE + exploration metrics from logged trajectories.

Exploration metrics are derived from the per-step agent [x,y,z] positions stored
in each record's `trajectory` (horizontal plane = x,z; y is up in Habitat):

  coverage_cells   distinct GRID_M-meter (x,z) cells visited  -> how much ground
  coverage_m2      coverage_cells * GRID_M^2
  path_length      cumulative path length (from trajectory_length)
  net_disp         straight-line start->final distance (x,z)
  wander_ratio     path_length / net_disp  (high = lots of wandering/backtrack)
  revisit_ratio    num_steps / coverage_cells  (high = revisiting same cells)

Degenerate flags: immediate_stop, mostly_spin (>90% of moves are turns),
hit_step_cap (>=490 steps).
"""
import json
import collections
import math
import argparse

GRID_M = 0.5  # cell size for coverage


def cell(pos):
    return (round(pos[0] / GRID_M), round(pos[2] / GRID_M))


def euclid_xz(a, b):
    return math.hypot(a[0] - b[0], a[2] - b[2])


def coverage(traj):
    pts = [s["position"] for s in traj if s.get("position")]
    if not pts:
        return 0, 0.0, 0.0
    cells = {cell(p) for p in pts}
    net = euclid_xz(pts[0], pts[-1])
    return len(cells), len(cells) * GRID_M * GRID_M, net


def degenerate(x):
    h = x["action_histogram"]
    moves = h["forward"] + h["turn_left"] + h["turn_right"]
    if x["num_steps"] >= 490:
        return "hit_step_cap"
    if x["num_steps"] <= 1 and h.get("stop", 0) >= 1:
        return "immediate_stop"
    if moves > 0 and (h["turn_left"] + h["turn_right"]) / moves > 0.9:
        return "mostly_spin"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="phase1_results.json")
    args = ap.parse_args()
    d = json.load(open(args.input))
    cells = collections.defaultdict(list)
    for x in d["results"]:
        cells[(x["pipeline"], x["condition"])].append(x)

    hdr = f"{'cell':30s} {'n':>2} {'SR':>5} {'SPL':>5} {'NE':>5} {'steps':>6} {'cover_m2':>9} {'wander':>7} {'revisit':>8}  degenerate"
    print(hdr)
    print("-" * len(hdr))
    for k in sorted(cells):
        v = cells[k]
        n = len(v)
        sr = sum(x["success"] for x in v) / n
        spl = sum(x["spl"] or 0 for x in v) / n
        ne = sum(x["distance_to_goal"] or 0 for x in v) / n
        steps = sum(x["num_steps"] for x in v) / n
        covs, wanders, revs = [], [], []
        for x in v:
            c, m2, net = coverage(x["trajectory"])
            covs.append(m2)
            pl = x["trajectory_length"] or 0
            if net > 0.3:
                wanders.append(pl / net)
            if c > 0:
                revs.append(x["num_steps"] / c)
        cover = sum(covs) / len(covs) if covs else 0
        wander = sum(wanders) / len(wanders) if wanders else 0
        rev = sum(revs) / len(revs) if revs else 0
        dg = collections.Counter(degenerate(x) for x in v if degenerate(x))
        name = f"{k[0]}/{k[1]}"
        print(f"{name:30s} {n:2d} {sr*100:4.0f}% {spl:5.2f} {ne:5.2f} {steps:6.1f} {cover:9.1f} {wander:7.2f} {rev:8.2f}  {dict(dg) or '-'}")

    # Per-episode detail for the key cell
    print("\n--- baseline/goal_only per-episode (sorted: successes first) ---")
    r = sorted(cells[("phase1_baseline", "goal_only")],
               key=lambda z: (not z["success"], int(z["episode_id"])))
    print(f"{'ep':>5} {'ok':>5} {'steps':>5} {'NE':>5} {'cover_m2':>9} {'wander':>7} {'flag':>12}")
    for x in r:
        c, m2, net = coverage(x["trajectory"])
        pl = x["trajectory_length"] or 0
        wd = pl / net if net > 0.3 else 0
        print(f"{x['episode_id']:>5} {str(x['success']):>5} {x['num_steps']:5d} {(x['distance_to_goal'] or 0):5.1f} {m2:9.1f} {wd:7.2f} {degenerate(x) or '':>12}")


if __name__ == "__main__":
    main()
