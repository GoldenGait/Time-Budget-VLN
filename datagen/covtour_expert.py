"""Phase 2 (alt) - semantic COVERAGE-TOUR expert (Layer 1).

Instead of emergent frontier search (which under-covers big scenes), precompute a small set of
viewpoints that SEE all navigable area (greedy set-cover), then visit them in RELEVANCE order:
each viewpoint scores P(target_category | its inferred room), where the room is voted from the
NON-target objects near it (leak-safe: uses the target *category*, never its position). Complete
coverage -> finds any reachable target. Tight = truncate after the first (most-relevant) region.

Consumes the same episode JSON as frontier_expert (sample_episodes_mp3d.py bakes object positions).

  python covtour_expert.py --episodes ep_mp3d.json --prior prior_table.json --out traces.json
"""
import argparse, collections, json, math, os
import numpy as np
import habitat_sim
import frontier_expert as fe

VIS_R = 2.0        # a viewpoint "sees" nav cells within this radius + grid-LoS
W_REL = 3.0        # tour weighting: prefer relevant rooms ...
W_TRAVEL = 0.20    # ... minus travel cost (geodesic), for a smooth human-like order


def compute_viewpoints(grid):
    """Greedy set-cover: fewest nav cells whose VIS_R + LoS discs cover all nav cells."""
    nav = [(int(i), int(j)) for i, j in grid.nav_cells]
    r = int(VIS_R / fe.CELL)
    cover = {}
    for (i, j) in nav:
        seen = set()
        for a in range(max(0, i - r), min(grid.nx, i + r + 1)):
            for b in range(max(0, j - r), min(grid.nz, j + r + 1)):
                if grid.nav[a, b] and (a - i) ** 2 + (b - j) ** 2 <= r * r and grid._los_grid(i, j, a, b):
                    seen.add((a, b))
        cover[(i, j)] = seen
    uncovered = set(nav); vps = []
    while uncovered:
        best = max(nav, key=lambda c: len(cover[c] & uncovered))
        gain = cover[best] & uncovered
        if not gain:
            break
        vps.append(best); uncovered -= gain
    return vps


def viewpoint_relevance(grid, vps, scene_objs, prior, cat):
    """relevance(vp) = P(cat | room(vp)); room voted from non-target objects near vp."""
    rel = {}
    for (i, j) in vps:
        wx, wz = grid.cw(i, j)
        votes = collections.Counter()
        for mp, ox, oz in scene_objs:
            if mp == cat:
                continue
            if math.hypot(ox - wx, oz - wz) <= fe.VOTE_R and mp in prior:
                for room, pr in prior[mp].items():
                    votes[room] += pr
        room = votes.most_common(1)[0][0] if votes else None
        rel[(i, j)] = prior.get(cat, {}).get(room, fe.EPS) if room else fe.EPS
    return rel


def tour_order(pf, grid, vps, rel, start_xz):
    """Relevance-weighted greedy order from the start: next = argmax(W_REL*rel - W_TRAVEL*geo)."""
    pts = []
    for (i, j) in vps:
        wx, wz = grid.cw(i, j); npt = pf.snap_point([wx, grid.fy, wz])
        if npt is not None and not np.any(np.isnan(npt)):
            pts.append(((i, j), np.asarray(npt, np.float32)))
    order = []; cur = np.array([start_xz[0], grid.fy, start_xz[1]], np.float32); remaining = pts[:]
    while remaining:
        def score(vp):
            g = fe.geo(pf, cur, vp[1])
            return (-1e9) if not math.isfinite(g) else (W_REL * rel[vp[0]] - W_TRAVEL * g)
        nxt = max(remaining, key=score); remaining.remove(nxt)
        if fe.geo(pf, cur, nxt[1]) == math.inf:            # unreachable island -> drop
            continue
        order.append(nxt); cur = nxt[1]
    return order


def run(sim, ep, targets, prior, scene_objs, vp_cache):
    pf = sim.pathfinder
    cat = ep["category"]; tgt = targets[cat]
    tgt_vps = np.asarray(tgt["viewpoints"], np.float32)
    tgt_obj = np.asarray(tgt["object_positions"], np.float32)
    fe.set_pose(sim, ep["start_position"], ep["start_yaw"])
    grid = fe.Grid(pf, ep["start_position"][1])
    key = round(ep["start_position"][1], 1)
    if key not in vp_cache:
        vp_cache[key] = compute_viewpoints(grid)           # coverage set-cover per floor (cached)
    vps = vp_cache[key]
    rel = viewpoint_relevance(grid, vps, scene_objs, prior, cat)
    order = tour_order(pf, grid, vps, rel, (ep["start_position"][0], ep["start_position"][2]))
    follower = habitat_sim.GreedyGeodesicFollower(pf, sim.get_agent(0), goal_radius=0.5,
                                                  forward_key="move_forward", left_key="turn_left", right_key="turn_right")
    prims, vp_ends, found, giveup = [], [], False, None

    def near_vp():
        p = fe.apos(sim)
        if len(tgt_vps) == 0:
            return math.inf
        de = np.hypot(tgt_vps[:, 0] - p[0], tgt_vps[:, 2] - p[2]); close = tgt_vps[de < 2.0]
        return min((fe.geo(pf, p, v) for v in close), default=math.inf)

    def detect():
        if near_vp() >= fe.SEE_DIST:
            return False
        p = fe.apos(sim); obj = min(tgt_obj, key=lambda o: (o[0] - p[0]) ** 2 + (o[2] - p[2]) ** 2)
        vx, vz = obj[0] - p[0], obj[2] - p[2]; n = math.hypot(vx, vz)
        if n < 1e-6:
            return True
        fx, fz = fe.fwd_xz(sim); return (vx * fx + vz * fz) / n >= math.cos(fe.HALF_FOV)

    def face():
        turns = []; p = fe.apos(sim); obj = min(tgt_obj, key=lambda o: (o[0] - p[0]) ** 2 + (o[2] - p[2]) ** 2)
        for _ in range(fe.FACE_CAP):
            p = fe.apos(sim); vx, vz = obj[0] - p[0], obj[2] - p[2]; n = math.hypot(vx, vz)
            if n < 1e-6:
                break
            fx, fz = fe.fwd_xz(sim)
            if (vx * fx + vz * fz) / n >= math.cos(fe.FACE_TOL):
                break
            act = "turn_right" if (fx * vz - fz * vx) > 0 else "turn_left"; sim.step(act); turns.append(act)
        return turns

    def look_around():
        for _ in range(fe.TURNS_360):
            if detect():
                return True
            sim.step("turn_left"); prims.append("turn_left")
        return detect()

    if detect() or look_around():
        prims += face(); found = True
    for (ij, npt) in order:
        if found:
            break
        follower.reset(); leg = 0
        while leg < fe.LEG_CAP:
            try:
                a = follower.next_action_along(npt)
            except habitat_sim.errors.GreedyFollowerError:
                break
            if a is None:
                break
            sim.step(a); prims.append(a); leg += 1
            if detect():
                prims += face(); found = True; break
        if found:
            break
        if look_around():                                  # survey the viewpoint's room
            prims += face(); found = True
        vp_ends.append(len(prims))
        if found:
            break
    if not found:
        giveup = "tour_exhausted"
    return {"category": cat, "start_position": ep["start_position"], "start_yaw": ep["start_yaw"],
            "actions": prims, "steps": len(prims), "found": bool(found), "giveup": giveup,
            "n_viewpoints": len(vps), "vp_ends": vp_ends, "T_loose": len(prims),
            "T_tight": vp_ends[0] if vp_ends else len(prims), "episode_id": ep["episode_id"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", default="datagen/ep_mp3d.json")
    ap.add_argument("--prior", default="datagen/prior_table.json")
    ap.add_argument("--out", default="datagen/traces_covtour.json")
    ap.add_argument("--scenes", nargs="+", default=None)
    ap.add_argument("--max-eps", type=int, default=0)
    ap.add_argument("--gpu", type=int, default=0)
    a = ap.parse_args()
    prior = json.load(open(a.prior))["prior"]
    data = json.load(open(a.episodes))
    out = {"meta": {"expert": "covtour", "vis_r": VIS_R, "w_rel": W_REL, "w_travel": W_TRAVEL}, "scenes": {}}
    tot = found = 0
    for stem, sc in sorted(data["scenes"].items()):
        if a.scenes and stem not in a.scenes:
            continue
        eps = sc["episodes"][: a.max_eps] if a.max_eps else sc["episodes"]
        if not eps:
            continue
        sim = fe.make_sim(stem, a.gpu, glb=sc.get("glb"))
        scene_objs = sc.get("objects", [])
        vp_cache = {}
        traces = []
        for ep in eps:
            import time as _t; t0 = _t.time()
            tr = run(sim, ep, sc["targets"], prior, scene_objs, vp_cache)
            traces.append(tr); tot += 1; found += tr["found"]
            print(f"    {stem} {ep['episode_id']:12s} found={tr['found']!s:5} T={tr['T_loose']:4d} "
                  f"vps={tr['n_viewpoints']} {_t.time()-t0:.1f}s", flush=True)
        out["scenes"][stem] = {"traces": traces}
        fr = sum(t["found"] for t in traces)
        print(f"[{stem}] {len(traces)} eps | found {fr}/{len(traces)}", flush=True)
        sim.close()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(out, open(a.out, "w"))
    print(f"\nTOTAL found {found}/{tot} ({found/max(1,tot):.0%}) -> {a.out}")


if __name__ == "__main__":
    main()
