"""Phase 1 — budget-conditioned multi-goal episode generation.

For one MP3D scene, sample (start_pose, G1, G2) and compute EMPIRICAL costs in
low-level primitive steps (0.25m forward / 15 deg turn) using habitat-sim's
GreedyGeodesicFollower as the expert. Goal reached = within SUCCESS_DISTANCE
(0.2 m) of the goal's navigable point. Records nearest_only and both_tour and
keeps episodes where a budget window can straddle the one-goal/both-goal flip.

Usage:
  python generate_episodes.py GdvgFV5R1Z5 --n 20 --out multigoal_episodes/data/episodes_GdvgFV5R1Z5.json
"""
import argparse
import json
import math
import os

import numpy as np
import habitat_sim
from habitat_sim.utils.common import quat_from_angle_axis

MP3D = "/media/maitree-tiamat/Expansion/NaVILA_data/scene_datasets/mp3d"

# action-space granularity (must match vlnce_task.yaml: FORWARD_STEP_SIZE / TURN_ANGLE)
FORWARD_M = 0.25
TURN_DEG = 15.0
SUCCESS_DISTANCE = 0.2      # object-goal radius (R2R's 3.0m is too coarse for small scenes)
MAX_LEG_STEPS = 500         # MAX_EPISODE_STEPS cap; a leg that exceeds it is "unreachable"

# goal-worthy mpcat40 categories (reconned whitelist)
WHITELIST = {"bed", "sofa", "toilet", "sink", "shower", "tv_monitor",
             "table", "counter", "cabinet", "chest_of_drawers", "stool", "chair"}


# ---------------------------------------------------------------- sim setup
def make_sim(scene_id):
    glb = f"{MP3D}/{scene_id}/{scene_id}.glb"
    backend = habitat_sim.SimulatorConfiguration()
    backend.scene_id = glb
    backend.enable_physics = False

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.action_space = {
        "move_forward": habitat_sim.agent.ActionSpec(
            "move_forward", habitat_sim.agent.ActuationSpec(amount=FORWARD_M)),
        "turn_left": habitat_sim.agent.ActionSpec(
            "turn_left", habitat_sim.agent.ActuationSpec(amount=TURN_DEG)),
        "turn_right": habitat_sim.agent.ActionSpec(
            "turn_right", habitat_sim.agent.ActuationSpec(amount=TURN_DEG)),
    }
    sim = habitat_sim.Simulator(habitat_sim.Configuration(backend, [agent_cfg]))
    return sim


# ---------------------------------------------------------------- geometry helpers
def geo(pf, a, b):
    """Geodesic distance between two points; inf if no path."""
    sp = habitat_sim.ShortestPath()
    sp.requested_start = np.asarray(a, dtype=np.float32)
    sp.requested_end = np.asarray(b, dtype=np.float32)
    return sp.geodesic_distance if pf.find_path(sp) else math.inf


def set_agent(sim, pos, yaw):
    st = habitat_sim.AgentState()
    st.position = np.asarray(pos, dtype=np.float32)
    st.rotation = quat_from_angle_axis(yaw, np.array([0.0, 1.0, 0.0]))
    sim.get_agent(0).set_state(st)


def agent_pos(sim):
    return sim.get_agent(0).get_state().position


def rollout_cost(sim, follower, goal_navpoint):
    """Drive from the agent's CURRENT pose to within SUCCESS_DISTANCE of
    goal_navpoint, counting low-level primitive steps and recording the action
    trace. Heading is whatever the agent currently has (so legs can be chained).
    Returns (steps, reached, actions, final_dist)."""
    pf = sim.pathfinder
    steps = 0
    actions = []
    while steps < MAX_LEG_STEPS:
        d = geo(pf, agent_pos(sim), goal_navpoint)
        if d <= SUCCESS_DISTANCE:
            return steps, True, actions, float(d)
        try:
            action = follower.next_action_along(goal_navpoint)
        except habitat_sim.errors.GreedyFollowerError:
            return steps, False, actions, float(geo(pf, agent_pos(sim), goal_navpoint))
        if action is None:          # follower's own (tight) radius reached
            return steps, True, actions, float(geo(pf, agent_pos(sim), goal_navpoint))
        sim.step(action)
        actions.append(action)
        steps += 1
    return steps, False, actions, float(geo(pf, agent_pos(sim), goal_navpoint))


# ---------------------------------------------------------------- object sampling
def collect_goal_objects(sim):
    """Snap each whitelist object to a navigable point on the main floor."""
    pf = sim.pathfinder
    floor_y = pf.get_random_navigable_point()[1]
    out = []
    for o in sim.semantic_scene.objects:
        if o is None or o.category is None:
            continue
        cat = o.category.name()
        if cat not in WHITELIST:
            continue
        c = np.array([o.aabb.center[0], o.aabb.center[1], o.aabb.center[2]], dtype=np.float32)
        if abs(c[1] - floor_y) > 1.5:        # keep this floor
            continue
        snap = pf.snap_point(c)
        if snap is None or np.any(np.isnan(snap)):
            continue
        out.append({"id": int(o.id.split("_")[-1]) if "_" in str(o.id) else o.id,
                    "category": cat,
                    "center": [float(x) for x in c],
                    "navpoint": [float(x) for x in snap]})
    return out, floor_y


# ---------------------------------------------------------------- episode generation
def make_episode(sim, follower, scene_id, goals, rng, min_ratio,
                 force_g1=None, force_g2=None):
    pf = sim.pathfinder
    start = pf.get_random_navigable_point()
    yaw = float(rng.uniform(-math.pi, math.pi))

    if force_g1 is not None:
        # caller pins the exact goal pair; just require both reachable from start
        g1, g2 = force_g1, force_g2
        if not (math.isfinite(geo(pf, start, g1["navpoint"]))
                and math.isfinite(geo(pf, start, g2["navpoint"]))):
            return None
    else:
        # pick two distinct-category goals reachable from start
        cand = [g for g in goals if math.isfinite(geo(pf, start, g["navpoint"]))]
        cats = sorted({g["category"] for g in cand})
        if len(cats) < 2:
            return None
        c1, c2 = rng.choice(cats, size=2, replace=False)
        g1 = rng.choice([g for g in cand if g["category"] == c1])
        g2 = rng.choice([g for g in cand if g["category"] == c2])
    p1, p2 = g1["navpoint"], g2["navpoint"]

    # order A: S -> G1 -> G2 (continuous, heading carried)
    set_agent(sim, start, yaw); follower.reset()
    a1, ok, a1_acts, a1_dist = rollout_cost(sim, follower, p1)
    if not ok:
        return None
    a2, a2_ok, a2_acts, a2_dist = rollout_cost(sim, follower, p2)
    orderA = a1 + a2 if a2_ok else math.inf

    # order B: S -> G2 -> G1
    set_agent(sim, start, yaw); follower.reset()
    b1, ok, b1_acts, b1_dist = rollout_cost(sim, follower, p2)
    if not ok:
        return None
    b2, b2_ok, b2_acts, b2_dist = rollout_cost(sim, follower, p1)
    orderB = b1 + b2 if b2_ok else math.inf

    both_tour = min(orderA, orderB)
    if not math.isfinite(both_tour):
        return None
    nearest_only = min(a1, b1)
    ordering = "G1,G2" if orderA <= orderB else "G2,G1"

    if nearest_only == 0 or both_tour / nearest_only < min_ratio:
        return None

    # tight regime: the fresh-heading leg to the NEARER goal, then stop
    if a1 <= b1:
        tight = {"target_slot": "G1", "target_category": g1["category"],
                 "actions": a1_acts, "steps": a1, "reached": True, "final_dist": a1_dist}
    else:
        tight = {"target_slot": "G2", "target_category": g2["category"],
                 "actions": b1_acts, "steps": b1, "reached": True, "final_dist": b1_dist}

    # loose regime: the cheaper continuous two-leg tour, then stop
    if orderA <= orderB:
        loose = {"order_slots": ["G1", "G2"],
                 "order_categories": [g1["category"], g2["category"]],
                 "actions": a1_acts + a2_acts, "leg_breaks": [len(a1_acts)],
                 "steps": a1 + a2, "reached": a2_ok, "final_dist": a2_dist}
    else:
        loose = {"order_slots": ["G2", "G1"],
                 "order_categories": [g2["category"], g1["category"]],
                 "actions": b1_acts + b2_acts, "leg_breaks": [len(b1_acts)],
                 "steps": b1 + b2, "reached": b2_ok, "final_dist": b2_dist}

    return {
        "scene": scene_id,
        "start_pose": {"position": [float(x) for x in start], "yaw": yaw},
        "G1": g1, "G2": g2,
        "segment_costs": {"S_G1": a1, "S_G2": b1},   # fresh-heading legs
        "nearest_only": int(nearest_only),
        "both_tour": int(both_tour),
        "ordering_for_both": ordering,
        "traces": {"tight": tight, "loose": loose},
    }


def histogram(name, vals):
    vals = np.array(vals)
    print(f"\n{name}: n={len(vals)} min={vals.min()} max={vals.max()} "
          f"mean={vals.mean():.1f} median={np.median(vals):.0f}")
    counts, edges = np.histogram(vals, bins=8)
    for c, lo, hi in zip(counts, edges[:-1], edges[1:]):
        print(f"  [{lo:5.0f},{hi:5.0f}) {'#' * c} {c}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--min-ratio", type=float, default=1.5)
    ap.add_argument("--max-tries", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", default="train",
                    help="train | val | test_heldout (whole-scene holdout)")
    ap.add_argument("--first-pair", default=None,
                    help="pin episode 0 to this goal-category pair, e.g. 'sink,toilet'")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    sim = make_sim(args.scene)
    sim.seed(args.seed)             # make navigable-point sampling reproducible too
    follower = habitat_sim.GreedyGeodesicFollower(
        sim.pathfinder, sim.get_agent(0), goal_radius=FORWARD_M,
        forward_key="move_forward", left_key="turn_left", right_key="turn_right")

    goals, floor_y = collect_goal_objects(sim)
    cats = sorted({g["category"] for g in goals})
    print(f"scene {args.scene}: {len(goals)} goal objects on floor y~{floor_y:.2f}; "
          f"categories={cats}")
    if len({g['category'] for g in goals}) < 2:
        print("not enough distinct goal categories; aborting"); return

    episodes, tries = [], 0

    # optionally pin episode 0 to a specific goal pair (deterministic search)
    if args.first_pair:
        c1, c2 = [s.strip() for s in args.first_pair.split(",")]
        fg1 = next((g for g in goals if g["category"] == c1), None)
        fg2 = next((g for g in goals if g["category"] == c2), None)
        if fg1 is None or fg2 is None:
            print(f"first-pair {args.first_pair!r} not both present; aborting"); return
        while not episodes and tries < args.max_tries:
            tries += 1
            ep = make_episode(sim, follower, args.scene, goals, rng, args.min_ratio,
                              force_g1=fg1, force_g2=fg2)
            if ep:
                episodes.append(ep)
        if not episodes:
            print(f"could not pin first-pair {args.first_pair!r}; aborting"); return

    while len(episodes) < args.n and tries < args.max_tries:
        tries += 1
        ep = make_episode(sim, follower, args.scene, goals, rng, args.min_ratio)
        if ep:
            episodes.append(ep)
    print(f"\nkept {len(episodes)}/{args.n} episodes in {tries} tries")

    # stamp identity/provenance now that ordering is fixed
    for i, ep in enumerate(episodes):
        ep["episode_id"] = f"{args.scene}_ep{i}"
        ep["split"] = args.split

    if episodes:
        histogram("nearest_only (steps)", [e["nearest_only"] for e in episodes])
        histogram("both_tour    (steps)", [e["both_tour"] for e in episodes])
        ratios = [e["both_tour"] / e["nearest_only"] for e in episodes]
        print(f"\nboth/nearest ratio: min={min(ratios):.2f} max={max(ratios):.2f} "
              f"mean={np.mean(ratios):.2f}")

    out = args.out or f"multigoal_episodes/data/episodes_{args.scene}.json"
    with open(out, "w") as f:
        json.dump({"scene": args.scene,
                   "seed": args.seed,
                   "split": args.split,
                   "action_unit": {"forward_m": FORWARD_M, "turn_deg": TURN_DEG,
                                   "success_distance": SUCCESS_DISTANCE},
                   "episodes": episodes}, f, indent=2)
    print(f"saved {len(episodes)} episodes -> {out}")
    sim.close()


if __name__ == "__main__":
    main()
