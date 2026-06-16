#!/usr/bin/env python3
"""Analyze Phase 0 results: compare NaVILA behavior across the three
instruction conditions (original / short budget / long budget)."""

import argparse
import json
from collections import defaultdict

CONDITION_ORDER = ["original", "short_budget", "long_budget"]
CONDITION_LABEL = {
    "original": "Original",
    "short_budget": "Short budget",
    "long_budget": "Long budget",
}
ACTION_KEYS = ["forward", "turn_left", "turn_right", "stop"]


def load(path):
    with open(path) as f:
        data = json.load(f)
    # index: episode_id -> condition -> record
    by_ep = defaultdict(dict)
    by_cond = defaultdict(list)
    for r in data["results"]:
        by_ep[r["episode_id"]][r["condition"]] = r
        by_cond[r["condition"]].append(r)
    return data, by_ep, by_cond


def mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else float("nan")


def print_summary_table(by_cond):
    print("=" * 72)
    print("(a) SUMMARY TABLE")
    print("=" * 72)
    header = f"{'Condition':<13}| {'SR':>6} | {'SPL':>6} | {'Avg Steps':>9} | {'Avg Traj Length':>15}"
    print(header)
    print("-" * len(header))
    for cond in CONDITION_ORDER:
        records = by_cond.get(cond, [])
        if not records:
            continue
        sr = mean([1.0 if r["success"] else 0.0 for r in records]) * 100
        spl = mean([r["spl"] for r in records])
        steps = mean([r["num_steps"] for r in records])
        traj = mean([r["trajectory_length"] for r in records])
        print(
            f"{CONDITION_LABEL[cond]:<13}| {sr:>5.1f}% | {spl:>6.3f} | "
            f"{steps:>9.1f} | {traj:>13.2f} m"
        )
    print()


def print_per_episode(by_ep):
    print("=" * 72)
    print("(b) PER-EPISODE TRAJECTORY LENGTH ACROSS CONDITIONS")
    print("=" * 72)
    header = (
        f"{'Episode':<10}| {'Original':>9} | {'Short':>9} | {'Long':>9} | {'Max Δ (m)':>9}"
    )
    print(header)
    print("-" * len(header))
    for ep_id in sorted(by_ep, key=lambda x: int(x) if str(x).isdigit() else x):
        conds = by_ep[ep_id]
        trajs = {c: conds[c]["trajectory_length"] for c in CONDITION_ORDER if c in conds}
        if not trajs:
            continue
        vals = [v for v in trajs.values() if v is not None]
        max_delta = (max(vals) - min(vals)) if len(vals) >= 2 else 0.0

        def fmt(c):
            v = trajs.get(c)
            return f"{v:>9.2f}" if v is not None else f"{'—':>9}"

        print(
            f"{ep_id:<10}| {fmt('original')} | {fmt('short_budget')} | "
            f"{fmt('long_budget')} | {max_delta:>9.2f}"
        )
    print()


def print_action_histograms(by_cond):
    print("=" * 72)
    print("(c) ACTION DISTRIBUTION ACROSS CONDITIONS")
    print("=" * 72)
    header = f"{'Condition':<13}|" + "".join(f" {k:>10} |" for k in ACTION_KEYS)
    print(header)
    print("-" * len(header))
    for cond in CONDITION_ORDER:
        records = by_cond.get(cond, [])
        if not records:
            continue
        totals = {k: sum(r["action_histogram"].get(k, 0) for r in records) for k in ACTION_KEYS}
        grand = sum(totals.values()) or 1
        cells = "".join(f" {totals[k]:>4} ({100*totals[k]/grand:4.1f}%)|" for k in ACTION_KEYS)
        print(f"{CONDITION_LABEL[cond]:<13}|{cells}")
    print()


def print_divergent(by_ep, traj_threshold=1.0):
    print("=" * 72)
    print(f"(d) EPISODES WHERE BEHAVIOR CHANGED (Δtraj > {traj_threshold} m or success flipped)")
    print("=" * 72)
    any_found = False
    for ep_id in sorted(by_ep, key=lambda x: int(x) if str(x).isdigit() else x):
        conds = by_ep[ep_id]
        present = [c for c in CONDITION_ORDER if c in conds]
        if len(present) < 2:
            continue
        trajs = [conds[c]["trajectory_length"] for c in present if conds[c]["trajectory_length"] is not None]
        successes = {conds[c]["success"] for c in present}
        traj_delta = (max(trajs) - min(trajs)) if len(trajs) >= 2 else 0.0
        success_changed = len(successes) > 1
        if traj_delta > traj_threshold or success_changed:
            any_found = True
            reasons = []
            if traj_delta > traj_threshold:
                reasons.append(f"Δtraj={traj_delta:.2f}m")
            if success_changed:
                reasons.append("success flipped")
            print(f"  episode {ep_id}: {', '.join(reasons)}")
            for c in present:
                r = conds[c]
                tl = r["trajectory_length"]
                tl_s = f"{tl:.2f}m" if tl is not None else "—"
                print(
                    f"      {CONDITION_LABEL[c]:<13} traj={tl_s:<8} "
                    f"steps={r['num_steps']:<4} success={r['success']} spl={r['spl']:.3f}"
                )
    if not any_found:
        print("  None — model behaved identically across all conditions on every episode.")
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="phase0_results.json")
    parser.add_argument("--traj-threshold", type=float, default=1.0)
    args = parser.parse_args()

    data, by_ep, by_cond = load(args.input)
    meta = data["metadata"]
    print()
    print(f"Phase 0 analysis — model={meta['model']} split={meta['split']} "
          f"seed={meta['seed']} episodes={meta['num_episodes']}")
    print(f"short_budget_text={meta['short_budget_text']!r}  "
          f"long_budget_text={meta['long_budget_text']!r}")
    print()

    print_summary_table(by_cond)
    print_per_episode(by_ep)
    print_action_histograms(by_cond)
    print_divergent(by_ep, args.traj_threshold)


if __name__ == "__main__":
    main()
