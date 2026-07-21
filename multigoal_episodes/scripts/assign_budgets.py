"""Layer 1.5 — decorrelated budget assignment for the tight vs explore contrastive set.

For each kept episode assign, in PRIMITIVE-STEP units, a narrow band STRADDLING the
episode's own flip point explore_len:
  explore budget = explore_len * (1 + U[gap, span])   (just above -> search feasible)
  tight   budget = explore_len * (1 - U[gap, span])   (just below -> explore infeasible),
                   clamped up to tight_len so the tight trace stays feasible.

Both budgets hug explore_len, which varies widely across episodes, so their MARGINAL
distributions overlap (a big-explore_len episode's tight budget exceeds a small one's
explore budget) -> NO single global cutoff K separates the two behaviours -> the model
must read the budget RELATIVE to the scene. Same guardrail as build_contrastive_pairs.
(A wide 'explore up to a shared ceiling' scheme was tried and FAILED the guardrail:
explore budgets skewed high, tight low, so a global cutoff separated them.)

Keep an episode only if the explore expert found the target AND explore_len > tight_len
(drops give-ups and degenerate found-at-spawn traces). Never touches test_heldout.

Usage:
  python assign_budgets.py --ep-dir multigoal_episodes/data --split train \
      --out /media/.../budget_vln_sft/budgets_train.json
"""
import argparse
import glob
import json
import os

import numpy as np


def best_global_cutoff_acc(budgets, labels):
    """labels: 1=explore, 0=tight. Accuracy of the best 'budget>=thr -> explore' rule."""
    budgets = np.asarray(budgets)
    labels = np.asarray(labels)
    best = 0.0
    best_thr = 0
    for thr in sorted(set(budgets.tolist())):
        acc = ((budgets >= thr) == (labels == 1)).mean()
        if acc > best:
            best, best_thr = acc, thr
    return best, best_thr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ep-dir", default="multigoal_episodes/data")
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", required=True)
    ap.add_argument("--gap", type=float, default=0.05, help="min fractional margin from explore_len")
    ap.add_argument("--span", type=float, default=0.35, help="max fractional margin from explore_len")
    ap.add_argument("--seed", type=int, default=10)
    args = ap.parse_args()
    assert 0 < args.gap < args.span < 1.0
    rng = np.random.default_rng(args.seed)

    # ---- gather kept episodes ----
    kept = []
    skipped = {"not_found": 0, "explore_le_tight": 0, "no_explore_trace": 0}
    for f in sorted(glob.glob(os.path.join(args.ep_dir, "episodes_*.json"))):
        d = json.load(open(f))
        for e in d["episodes"]:
            if e.get("split", d.get("split")) != args.split:
                continue
            tr = e["traces"]
            if "explore" not in tr:
                skipped["no_explore_trace"] += 1
                continue
            if not tr["explore"]["found"]:
                skipped["not_found"] += 1
                continue
            t_len = tr["tight"]["steps"]
            e_len = tr["explore"]["steps"]
            if e_len <= t_len:
                skipped["explore_le_tight"] += 1
                continue
            kept.append({"episode_id": e["episode_id"], "scene": e["scene"],
                         "tight_len": int(t_len), "explore_len": int(e_len),
                         "category": tr["explore"]["target_category"]})

    if not kept:
        print("no kept episodes; did you run generate_frontier_episodes.py first?")
        return

    # ---- assign budgets (narrow band straddling each episode's explore_len) ----
    budgets = {}
    b_tight, b_explore = [], []
    for k in kept:
        t_len, e_len = k["tight_len"], k["explore_len"]
        eb = max(e_len + 1, round(e_len * (1.0 + rng.uniform(args.gap, args.span))))   # > explore_len
        tb = round(e_len * (1.0 - rng.uniform(args.gap, args.span)))                   # < explore_len
        tb = max(tb, t_len)                                  # feasible for the tight trace
        budgets[k["episode_id"]] = {
            "tight_budget": tb, "explore_budget": eb,
            "tight_len": t_len, "explore_len": e_len,
            "category": k["category"], "scene": k["scene"],
        }
        b_tight.append(tb)
        b_explore.append(eb)

    # ---- guardrail: no global cutoff separates tight from explore ----
    allb = b_tight + b_explore
    labels = [0] * len(b_tight) + [1] * len(b_explore)
    acc, thr = best_global_cutoff_acc(allb, labels)
    verdict = ("PASS: no global cutoff separates -> budget must be read relative to scene"
               if acc < 0.65 else
               "FAIL: a global cutoff still separates -> widen the band (raise --span)")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump({"split": args.split, "gap": args.gap, "span": args.span, "budgets": budgets},
              open(args.out, "w"), indent=2)

    bt, be = np.array(b_tight), np.array(b_explore)
    print(f"kept {len(kept)} episodes ({args.split}); skipped {skipped}")
    print(f"band: explore_len * (1 +/- U[{args.gap},{args.span}])")
    print(f"tight   budget: min={bt.min()} max={bt.max()} mean={bt.mean():.0f}")
    print(f"explore budget: min={be.min()} max={be.max()} mean={be.mean():.0f}")
    print(f"overlap: {(bt.max() >= be.min())}  (tight_max={bt.max()} >= explore_min={be.min()})")
    print(f"[GUARDRAIL] best global budget-cutoff acc = {acc:.3f} (thr>={thr}); chance=0.500")
    print(f"[GUARDRAIL] {verdict}")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
