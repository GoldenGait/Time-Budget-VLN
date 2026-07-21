"""Option A demo — coverage-tour explorer.

Instead of emergent frontier search, precompute a small set of viewpoints that
together SEE all navigable area (greedy set-cover), order them into a short
nearest-neighbour tour from the start, and walk it; stop when the target comes
within SEE_R + line-of-sight (turn to face it first). Target-agnostic: the tour
is built without the target; the target enters only at the stop check.

Guarantees bounded-length COMPLETE coverage -> reliably finds any reachable target.

  python coverage_tour_demo.py 17DRP5sb8fy --episode 0 --viz
  python coverage_tour_demo.py 1LXtFkjw3qL --all
"""
import argparse, json, math, os
from collections import deque
import numpy as np
import habitat_sim
from habitat_sim.utils.common import quat_from_angle_axis
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

MP3D = "/media/maitree-tiamat/Expansion/NaVILA_data/scene_datasets/mp3d"
FORWARD_M, TURN_DEG = 0.25, 15.0
CELL = 0.5
VIS_R = 2.0          # a viewpoint "sees" nav cells within this radius + LoS (== SEE_R)
SEE_R = 2.0          # target found within this geodesic distance + LoS
HALF_FOV = math.radians(45.0)
FACE_TOL = math.radians(TURN_DEG / 2.0)  # end facing the target: center it within half a turn-step
LEG_CAP = 400
TURNS_360 = int(round(360.0 / TURN_DEG))


def make_sim(scene, gpu=1):
    b = habitat_sim.SimulatorConfiguration()
    b.scene_id = f"{MP3D}/{scene}/{scene}.glb"; b.enable_physics = False; b.gpu_device_id = gpu
    ac = habitat_sim.agent.AgentConfiguration()
    ac.action_space = {
        "move_forward": habitat_sim.agent.ActionSpec("move_forward", habitat_sim.agent.ActuationSpec(amount=FORWARD_M)),
        "turn_left": habitat_sim.agent.ActionSpec("turn_left", habitat_sim.agent.ActuationSpec(amount=TURN_DEG)),
        "turn_right": habitat_sim.agent.ActionSpec("turn_right", habitat_sim.agent.ActuationSpec(amount=TURN_DEG)),
    }
    return habitat_sim.Simulator(habitat_sim.Configuration(b, [ac]))


def set_agent(sim, pos, yaw):
    st = habitat_sim.AgentState(); st.position = np.asarray(pos, np.float32)
    st.rotation = quat_from_angle_axis(float(yaw), np.array([0., 1., 0.])); sim.get_agent(0).set_state(st)


def apos(sim):
    return np.asarray(sim.get_agent(0).get_state().position, np.float32)


def fwd_xz(sim):
    q = sim.get_agent(0).get_state().rotation
    th = 2.0 * math.atan2(q.y, q.w); return -math.sin(th), -math.cos(th)


def geo(pf, a, b):
    sp = habitat_sim.ShortestPath(); sp.requested_start = np.asarray(a, np.float32); sp.requested_end = np.asarray(b, np.float32)
    return sp.geodesic_distance if pf.find_path(sp) else math.inf


def los(pf, a, b, fy):
    ax, az, bx, bz = a[0], a[2], b[0], b[2]
    n = max(2, int(math.hypot(bx - ax, bz - az) / 0.15))
    return all(pf.is_navigable([ax + (bx - ax) * t, fy, az + (bz - az) * t]) for t in np.linspace(0, 1, n))


class Grid:
    def __init__(self, pf, fy):
        lo, _ = pf.get_bounds(); self.x0, self.z0, self.fy = float(lo[0]), float(lo[2]), float(fy)
        self.nav = np.ascontiguousarray(pf.get_topdown_view(CELL, self.fy).T)
        self.nx, self.nz = self.nav.shape

    def cw(self, i, j): return self.x0 + (i + .5) * CELL, self.z0 + (j + .5) * CELL
    def wc(self, x, z): return int((x - self.x0) / CELL), int((z - self.z0) / CELL)

    def clear(self, ci, cj, i, j):
        n = max(abs(i - ci), abs(j - cj))
        return all(self.nav[ci + (i - ci) * s // n, cj + (j - cj) * s // n] for s in range(1, n)) if n else True


def compute_viewpoints(grid):
    """Greedy set-cover: fewest nav cells whose VIS_R+LoS discs cover all nav cells."""
    navcells = [(i, j) for i in range(grid.nx) for j in range(grid.nz) if grid.nav[i, j]]
    r = int(VIS_R / CELL)
    cover = {}
    for (i, j) in navcells:
        seen = set()
        for a in range(max(0, i - r), min(grid.nx, i + r + 1)):
            for b in range(max(0, j - r), min(grid.nz, j + r + 1)):
                if grid.nav[a, b] and (a - i) ** 2 + (b - j) ** 2 <= r * r and grid.clear(i, j, a, b):
                    seen.add((a, b))
        cover[(i, j)] = seen
    uncovered = set(navcells); vps = []
    while uncovered:
        best = max(navcells, key=lambda c: len(cover[c] & uncovered))
        gain = cover[best] & uncovered
        if not gain: break
        vps.append(best); uncovered -= gain
    return vps, len(navcells)


def tour_order(pf, grid, vps, start):
    """Nearest-neighbour tour from the start pose over the viewpoints (geodesic)."""
    pts = [pf.snap_point([*grid.cw(i, j)[:1], grid.fy, grid.cw(i, j)[1]]) for (i, j) in vps]
    pts = [(vps[k], np.asarray(p, np.float32)) for k, p in enumerate(pts) if p is not None and not np.any(np.isnan(p))]
    order, cur, remaining = [], np.asarray(start, np.float32), pts[:]
    while remaining:
        remaining.sort(key=lambda vp: geo(pf, cur, vp[1]))
        nxt = remaining.pop(0); order.append(nxt); cur = nxt[1]
    return order


def detect(sim, pf, target, fy):
    ap = apos(sim); d = geo(pf, ap, target["navpoint"])
    if d > SEE_R: return False
    if d > 1.0 and not los(pf, ap, target["navpoint"], fy): return False
    return True


def turn_to_face(sim, target, prims):
    """Center the target within FACE_TOL of forward (not just inside the wide
    detect cone) so the final stop frame has the agent facing the goal."""
    for _ in range(TURNS_360):
        ap = apos(sim); vx, vz = target["center"][0] - ap[0], target["center"][2] - ap[2]
        n = math.hypot(vx, vz)
        if n < 1e-6: return
        fx, fz = fwd_xz(sim)
        if (vx * fx + vz * fz) / n >= math.cos(FACE_TOL): return
        act = "turn_right" if (fx * vz - fz * vx) > 0 else "turn_left"
        sim.step(act); prims.append(act)


def run(sim, target, start_pose, vps_cache):
    pf = sim.pathfinder
    follower = habitat_sim.GreedyGeodesicFollower(pf, sim.get_agent(0), goal_radius=FORWARD_M,
                                                  forward_key="move_forward", left_key="turn_left", right_key="turn_right")
    set_agent(sim, start_pose["position"], start_pose["yaw"]); follower.reset()
    grid = vps_cache["grid"]
    order = tour_order(pf, grid, vps_cache["vps"], start_pose["position"])
    prims, traj, found = [], [apos(sim)[[0, 2]].tolist()], False
    if detect(sim, pf, target, grid.fy):
        turn_to_face(sim, target, prims); found = True
    for (_, goal) in order:
        if found: break
        follower.reset(); leg = 0
        while leg < LEG_CAP:
            try:
                a = follower.next_action_along(goal)
            except habitat_sim.errors.GreedyFollowerError:
                break
            if a is None: break
            sim.step(a); prims.append(a); leg += 1
            traj.append(apos(sim)[[0, 2]].tolist())
            if detect(sim, pf, target, grid.fy):
                turn_to_face(sim, target, prims); found = True; break
    return prims, found, traj, order


def viz(grid, vps, order, traj, target, start, path, title):
    img = np.full((grid.nx, grid.nz, 3), .12); img[grid.nav] = [.30, .30, .34]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.imshow(np.transpose(img, (1, 0, 2)), origin="lower",
              extent=[grid.x0, grid.x0 + grid.nx * CELL, grid.z0, grid.z0 + grid.nz * CELL])
    vp_w = [grid.cw(i, j) for (i, j) in vps]
    ax.scatter([w[0] for w in vp_w], [w[1] for w in vp_w], c="orange", s=40, marker="s", label="coverage viewpoints", zorder=3)
    ow = [grid.cw(*vp[0]) for vp in order]
    ax.plot([w[0] for w in ow], [w[1] for w in ow], "--", color="orange", lw=1, alpha=.6, label="tour order")
    t = np.array(traj); ax.plot(t[:, 0], t[:, 1], "-", color="#1f77b4", lw=1.4, label="trajectory")
    ax.plot(start["position"][0], start["position"][2], "o", color="cyan", ms=11, label="start", zorder=4)
    ax.plot(target["navpoint"][0], target["navpoint"][2], "*", color="gold", ms=22, markeredgecolor="k",
            label=f"target ({target['category']})", zorder=5)
    ax.set_title(title, fontsize=10); ax.legend(loc="upper right", fontsize=8); ax.set_aspect("equal")
    fig.savefig(path, dpi=110, bbox_inches="tight"); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("scene"); ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--all", action="store_true"); ap.add_argument("--viz", action="store_true")
    ap.add_argument("--out", default="/tmp/claude-1013/-home-maitree-tiamat-Time-Budget-VLN/505270d0-b7a6-4273-8f99-9071cbeda526/scratchpad")
    a = ap.parse_args()
    eps = json.load(open(f"/home/maitree-tiamat/Time-Budget-VLN/multigoal_episodes/data/episodes_{a.scene}.json"))["episodes"]
    sim = make_sim(a.scene)
    todo = eps if a.all else [eps[a.episode]]
    found_n, ratios = 0, []
    sf_found = sf_tot = 0
    skipped_xfloor = 0
    for ep in todo:
        tgt = ep[ep["traces"]["tight"]["target_slot"]]
        dy = abs(tgt["navpoint"][1] - ep["start_pose"]["position"][1]); same_floor = dy < 0.8
        if not same_floor:
            skipped_xfloor += 1; continue                      # single-floor coverage can't reach it
        grid = Grid(sim.pathfinder, ep["start_pose"]["position"][1])   # per-episode floor
        vps, _ = compute_viewpoints(grid)
        cache = {"grid": grid, "vps": vps}
        prims, found, traj, order = run(sim, tgt, ep["start_pose"], cache)
        found_n += found; tight = ep["traces"]["tight"]["steps"]; ratios.append(len(prims) / max(1, tight))
        if same_floor:
            sf_tot += 1; sf_found += found
        print(f"  {ep['episode_id']:>22} {tgt['category']:<16} tight={tight:>3} cover={len(prims):>4} "
              f"found={str(found):<5} ratio={len(prims)/max(1,tight):>5.2f}x", flush=True)
        if a.viz and not a.all:
            p = os.path.join(a.out, f"covtour_{ep['episode_id']}.png")
            viz(grid, vps, order, traj, tgt, ep["start_pose"], p,
                f"COVERAGE TOUR {ep['episode_id']} {tgt['category']} | {len(vps)} viewpoints cover={len(prims)} tight={tight} found={found}")
            print(f"  viz -> {p}", flush=True)
    if a.all:
        r = np.array(ratios)
        print(f"\nSAME-FLOOR FOUND {sf_found}/{sf_tot} ({sf_found/max(1,sf_tot):.0%}) | "
              f"cross-floor skipped {skipped_xfloor} | cover/tight ratio med={np.median(r):.1f}x")
    sim.close()


if __name__ == "__main__":
    main()
