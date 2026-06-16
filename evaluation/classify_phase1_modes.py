#!/usr/bin/env python3
"""Phase 1 behavioral mode classifier for the goal_only cells.

For each goal_only rollout, derive geometry from the logged trajectory and bucket
it into a behavior mode:

  SUCCESS          reached the goal
  WANDER(revisit)  hit the step cap, churning the same ground (path >> covered area)
  WANDER(cover)    hit the step cap, moved around but never recognized/stopped
  SPIN             turned in place (>85% turns, ~no displacement)
  EARLY_QUIT       issued STOP but ended >3 m from goal
  OTHER            none of the above

Geometry (horizontal plane x,z; y is up in Habitat):
  disp    straight-line start->final distance
  path    cumulative path length
  detour  path / disp (1.0 = straight; high = backtracking)
  cov     distinct 0.5 m cells visited
  turn%   fraction of moves that are turns
"""
import json
import math
import argparse
import collections


def xz(p):
    return (p[0], p[2])


def classify(x):
    traj = [t["position"] if isinstance(t, dict) else t for t in x["trajectory"]]
    pts = [xz(p) for p in traj if p]
    path = sum(math.dist(pts[i], pts[i + 1]) for i in range(len(pts) - 1)) if len(pts) > 1 else 0.0
    disp = math.dist(pts[0], pts[-1]) if len(pts) > 1 else 0.0
    cov = len({(round(a / 0.5), round(b / 0.5)) for a, b in pts})
    h = x["action_histogram"]
    mv = h["forward"] + h["turn_left"] + h["turn_right"]
    turnfrac = (h["turn_left"] + h["turn_right"]) / max(mv, 1)
    detour = path / max(disp, 1e-3)
    ne = x["distance_to_goal"] or 0
    if x["success"]:
        mode = "SUCCESS"
    elif x["num_steps"] >= 490:
        mode = "WANDER(revisit)" if path / max(cov * 0.5, 1e-3) > 2.5 else "WANDER(cover)"
    elif turnfrac > 0.85 and disp < 1.0:
        mode = "SPIN"
    elif h["stop"] > 0 and ne > 3:
        mode = "EARLY_QUIT"
    else:
        mode = "OTHER"
    return dict(mode=mode, steps=x["num_steps"], disp=disp, path=path,
               detour=detour, cov=cov, turnfrac=turnfrac, ne=ne)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="phase1_results.json")
    args = ap.parse_args()
    d = json.load(open(args.input))

    # Iterate over every pipeline present in the results (auto-includes new cells).
    pipelines = sorted({x["pipeline"] for x in d["results"] if x["condition"] == "goal_only"})
    by = {}
    for cell in pipelines:
        rows = [x for x in d["results"] if x["pipeline"] == cell and x["condition"] == "goal_only"]
        print(f"\n=== {cell} / goal_only ===")
        by[cell] = {}
        for x in sorted(rows, key=lambda z: int(z["episode_id"])):
            c = classify(x)
            by[cell][x["episode_id"]] = c
            print(f"  ep {x['episode_id']:5s} {c['mode']:16s} steps={c['steps']:3d} disp={c['disp']:4.1f} "
                  f"path={c['path']:5.1f} detour={c['detour']:4.1f} cov={c['cov']:3d} turn%={c['turnfrac']:.2f} NE={c['ne']:.1f}")
        cnt = collections.Counter(v["mode"] for v in by[cell].values())
        print(f"  -- mode counts: {dict(cnt)}")

    # Success composition shift (pairwise vs the baseline pipeline, if present).
    ref = "phase1_baseline"
    if ref in by:
        b = {e for e, c in by[ref].items() if c["mode"] == "SUCCESS"}
        for cell in pipelines:
            if cell == ref:
                continue
            x = {e for e, c in by[cell].items() if c["mode"] == "SUCCESS"}
            print(f"\n=== success composition: {ref} vs {cell} ===")
            print(f"  {ref:26s} successes ({len(b)}): {sorted(b, key=int)}")
            print(f"  {cell:26s} successes ({len(x)}): {sorted(x, key=int)}")
            print(f"  gained: {sorted(x - b, key=int)}    lost: {sorted(b - x, key=int)}")

    # Attribution view: track the diagnostic episodes across ALL pipelines.
    BREAKS = ["53", "63", "464"]      # clean approaches lost under explore
    RESCUES = ["60", "200", "1187"]   # explore-needers gained under explore
    print("\n=== attribution: mode of diagnostic episodes per pipeline ===")
    hdr = "  {:8s}".format("ep") + "".join(f"{p.replace('phase1_',''):>20s}" for p in pipelines)
    print(hdr)
    for label, eps in (("BREAKS", BREAKS), ("RESCUES", RESCUES)):
        print(f"  -- {label} --")
        for e in eps:
            row = "  {:8s}".format(e) + "".join(
                f"{(by[p].get(e, {}).get('mode', '-')):>20s}" for p in pipelines
            )
            print(row)


if __name__ == "__main__":
    main()
