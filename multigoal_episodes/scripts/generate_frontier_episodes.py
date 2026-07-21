"""Layer 1 (explore) — frontier-coverage exploration expert for budget-VLN.

The loose-budget / "explore" regime: given only the target object, explore
TARGET-AGNOSTICALLY until it comes into view, then stop. Reuses each tight
episode's (start_pose, tight target) so tight (shortest path) and explore
(frontier search) form contrastive pairs over identical (scene, start, target).

Stop = geometric visibility (the ONLY place the target location is used): within
SEE_R geodesic, the object's centre inside the forward FOV cone, and line-of-sight
clear. No camera needed -> CPU only. Writes traces["explore"] into the episodes
file in place (or --out elsewhere).

Usage:
  python generate_frontier_episodes.py 17DRP5sb8fy
  python generate_frontier_episodes.py 17DRP5sb8fy --out /tmp/preview.json
"""
import argparse
import json
import math
import time
from collections import deque

import numpy as np
import habitat_sim
from habitat_sim.utils.common import quat_from_angle_axis

MP3D = "/media/maitree-tiamat/Expansion/NaVILA_data/scene_datasets/mp3d"

# must match the tight generator / vlnce_task.yaml action granularity
FORWARD_M, TURN_DEG = 0.25, 15.0
CELL = 0.5                       # occupancy cell size (coarser = faster frontiers)
REVEAL_R = 2.0                   # FOV reveal radius, <= SEE_R (no reveal-away w/o a detectable look)
SEE_R = 2.0                      # target "found" within this geodesic distance...
HALF_FOV = math.radians(45.0)    # ...and inside this half-FOV cone (camera hfov ~90deg)
FACE_TOL = math.radians(TURN_DEG / 2.0)  # end facing the target: center it within half a turn-step
LEG_CAP = 300                    # per-frontier-leg follower cap (anti-stuck, not behavioural)
SAFETY_STEPS = 3000              # total-primitive safety valve (bug backstop; logged if hit)
SCAN_EVERY = 12                  # mid-leg 360deg glance every N forward steps (~3m travel)
UNKNOWN, FREE, OCC = -1, 0, 1
TURNS_360 = int(round(360.0 / TURN_DEG))


# ---------------------------------------------------------------- sim (no sensors)
def make_sim(scene, gpu=0):
    glb = f"{MP3D}/{scene}/{scene}.glb"
    b = habitat_sim.SimulatorConfiguration()
    b.scene_id = glb
    b.enable_physics = False
    b.gpu_device_id = gpu
    ac = habitat_sim.agent.AgentConfiguration()
    ac.action_space = {
        "move_forward": habitat_sim.agent.ActionSpec("move_forward", habitat_sim.agent.ActuationSpec(amount=FORWARD_M)),
        "turn_left": habitat_sim.agent.ActionSpec("turn_left", habitat_sim.agent.ActuationSpec(amount=TURN_DEG)),
        "turn_right": habitat_sim.agent.ActionSpec("turn_right", habitat_sim.agent.ActuationSpec(amount=TURN_DEG)),
    }
    return habitat_sim.Simulator(habitat_sim.Configuration(b, [ac]))


def set_agent(sim, pos, yaw):
    st = habitat_sim.AgentState()
    st.position = np.asarray(pos, dtype=np.float32)
    st.rotation = quat_from_angle_axis(float(yaw), np.array([0.0, 1.0, 0.0]))
    sim.get_agent(0).set_state(st)


def apos(sim):
    return np.asarray(sim.get_agent(0).get_state().position, dtype=np.float32)


def fwd_xz(sim):
    """Forward in XZ. All rotations here are pure-Y, so yaw = 2*atan2(y,w)."""
    q = sim.get_agent(0).get_state().rotation
    theta = 2.0 * math.atan2(q.y, q.w)
    return -math.sin(theta), -math.cos(theta)


def turn_to_face(sim, target, prims):
    """Turn in place until the target center is within FACE_TOL of the camera
    forward axis, so the final (stop) frame has the agent facing the goal.
    Appends the turn primitives to `prims`; caps at one full revolution."""
    for _ in range(TURNS_360):
        ap = apos(sim)
        vx, vz = target["center"][0] - ap[0], target["center"][2] - ap[2]
        n = math.hypot(vx, vz)
        if n < 1e-6:
            return
        fx, fz = fwd_xz(sim)
        if (vx * fx + vz * fz) / n >= math.cos(FACE_TOL):
            return
        act = "turn_right" if (fx * vz - fz * vx) > 0 else "turn_left"
        sim.step(act)
        prims.append(act)


def geo(pf, a, b):
    sp = habitat_sim.ShortestPath()
    sp.requested_start = np.asarray(a, dtype=np.float32)
    sp.requested_end = np.asarray(b, dtype=np.float32)
    return sp.geodesic_distance if pf.find_path(sp) else math.inf


# ---------------------------------------------------------------- occupancy grid
class Grid:
    """Occupancy over the scene footprint. i indexes X, j indexes Z."""
    def __init__(self, pf, floor_y):
        lo, _ = pf.get_bounds()
        self.x0, self.z0, self.floor_y = float(lo[0]), float(lo[2]), float(floor_y)
        tdv = pf.get_topdown_view(CELL, self.floor_y)     # (nz, nx) bool, navigable
        self.nav = np.ascontiguousarray(tdv.T)            # -> (nx, nz)
        self.nx, self.nz = self.nav.shape
        self.seen = np.full((self.nx, self.nz), UNKNOWN, np.int8)

    def cw(self, i, j):
        return self.x0 + (i + 0.5) * CELL, self.z0 + (j + 0.5) * CELL

    def wc(self, x, z):
        return int((x - self.x0) / CELL), int((z - self.z0) / CELL)

    def _clear(self, ci, cj, i, j):
        n = max(abs(i - ci), abs(j - cj))
        if n == 0:
            return True
        for s in range(1, n):
            if not self.nav[ci + (i - ci) * s // n, cj + (j - cj) * s // n]:
                return False
        return True

    def reveal(self, pos, fwd):
        """Mark cells in the forward FOV cone within REVEAL_R with clear LoS."""
        px, pz = float(pos[0]), float(pos[2])
        fx, fz = fwd
        ci, cj = self.wc(px, pz)
        r = int(REVEAL_R / CELL) + 1
        cos_fov = math.cos(HALF_FOV)
        if self.nav[ci, cj]:
            self.seen[ci, cj] = FREE
        for i in range(max(0, ci - r), min(self.nx, ci + r + 1)):
            for j in range(max(0, cj - r), min(self.nz, cj + r + 1)):
                if self.seen[i, j] != UNKNOWN:
                    continue
                wx, wz = self.cw(i, j)
                vx, vz = wx - px, wz - pz
                d2 = vx * vx + vz * vz
                if d2 > REVEAL_R ** 2 or d2 < 1e-6:
                    continue
                if (vx * fx + vz * fz) < cos_fov * math.sqrt(d2):
                    continue
                if self._clear(ci, cj, i, j):
                    self.seen[i, j] = FREE if self.nav[i, j] else OCC

    def frontiers(self):
        fr = []
        for i in range(self.nx):
            for j in range(self.nz):
                if self.seen[i, j] != FREE:
                    continue
                hit = False
                for di in (-1, 0, 1):
                    for dj in (-1, 0, 1):
                        a, b = i + di, j + dj
                        if 0 <= a < self.nx and 0 <= b < self.nz \
                                and self.seen[a, b] == UNKNOWN and self.nav[a, b]:
                            hit = True
                            break
                    if hit:
                        break
                if hit:
                    fr.append((i, j))
        return fr

    def cluster(self, cells):
        cs = set(cells)
        out = []
        while cs:
            comp = [cs.pop()]
            dq = deque(comp)
            while dq:
                i, j = dq.popleft()
                for di in (-1, 0, 1):
                    for dj in (-1, 0, 1):
                        c = (i + di, j + dj)
                        if c in cs:
                            cs.discard(c)
                            comp.append(c)
                            dq.append(c)
            out.append(comp)
        return out


# ---------------------------------------------------------------- detection (geometric)
def los_clear(pf, a, b, floor_y):
    ax, az = float(a[0]), float(a[2])
    bx, bz = float(b[0]), float(b[2])
    n = max(2, int(math.hypot(bx - ax, bz - az) / 0.15))
    for t in np.linspace(0.0, 1.0, n):
        if not pf.is_navigable([ax + (bx - ax) * t, floor_y, az + (bz - az) * t]):
            return False
    return True


def detect(sim, pf, target, floor_y):
    """Geometric visibility: within SEE_R, centre in the forward FOV cone, LoS clear."""
    ap = apos(sim)
    d = geo(pf, ap, target["navpoint"])
    if d > SEE_R:
        return False, d
    cx, cz = target["center"][0], target["center"][2]
    vx, vz = cx - ap[0], cz - ap[2]
    n = math.hypot(vx, vz)
    if n > 1e-6:
        fx, fz = fwd_xz(sim)
        if (vx * fx + vz * fz) / n < math.cos(HALF_FOV):
            return False, d
    if d > 1.0 and not los_clear(pf, ap, target["navpoint"], floor_y):
        return False, d
    return True, d


def look_around(sim, pf, target, grid, prims):
    for _ in range(TURNS_360):
        if detect(sim, pf, target, grid.floor_y)[0]:
            return True
        sim.step("turn_left")
        prims.append("turn_left")
        grid.reveal(apos(sim), fwd_xz(sim))
    return detect(sim, pf, target, grid.floor_y)[0]


# ---------------------------------------------------------------- explore one episode
def explore(sim, target, start_pose):
    pf = sim.pathfinder
    follower = habitat_sim.GreedyGeodesicFollower(
        pf, sim.get_agent(0), goal_radius=FORWARD_M,
        forward_key="move_forward", left_key="turn_left", right_key="turn_right")
    set_agent(sim, start_pose["position"], start_pose["yaw"])
    follower.reset()
    grid = Grid(pf, start_pose["position"][1])
    grid.reveal(apos(sim), fwd_xz(sim))

    prims, choices = [], []
    found = False
    gave_up = None
    total = 0

    if look_around(sim, pf, target, grid, prims):
        found = True
    while not found:
        if detect(sim, pf, target, grid.floor_y)[0]:
            found = True
            break
        fr = grid.frontiers()
        if not fr:
            gave_up = "frontiers_exhausted"
            break
        p = apos(sim)
        ranked = []
        for k, comp in enumerate(grid.cluster(fr)):
            ci = int(round(np.mean([c[0] for c in comp])))
            cj = int(round(np.mean([c[1] for c in comp])))
            wx, wz = grid.cw(ci, cj)
            npt = pf.snap_point([wx, grid.floor_y, wz])
            if npt is None or np.any(np.isnan(npt)):
                continue
            ranked.append((geo(pf, p, npt), k, npt, (wx, wz)))
        if not ranked:
            gave_up = "no_reachable_frontier"
            break
        ranked.sort(key=lambda r: (r[0], r[1]))
        _, _, goal_pt, goal_xz = ranked[0]
        choices.append([float(goal_xz[0]), float(goal_xz[1])])

        follower.reset()
        leg = 0
        while leg < LEG_CAP and total < SAFETY_STEPS:
            try:
                a = follower.next_action_along(np.asarray(goal_pt, dtype=np.float32))
            except habitat_sim.errors.GreedyFollowerError:
                break
            if a is None:
                break
            sim.step(a)
            prims.append(a)
            leg += 1
            total += 1
            grid.reveal(apos(sim), fwd_xz(sim))
            if detect(sim, pf, target, grid.floor_y)[0]:
                found = True
                break
            if a == "move_forward" and leg % SCAN_EVERY == 0:
                total += TURNS_360
                if look_around(sim, pf, target, grid, prims):
                    found = True
                    break
        if found:
            break
        if total >= SAFETY_STEPS:
            gave_up = "safety_cap"
            break
        total += TURNS_360
        if look_around(sim, pf, target, grid, prims):
            found = True
            break

    if found:
        turn_to_face(sim, target, prims)        # end the trace facing the goal

    final_dist = float(geo(pf, apos(sim), target["navpoint"]))
    return {
        "actions": prims,                       # primitives; flattener appends the stop
        "steps": len(prims),
        "found": bool(found),
        "final_dist": final_dist,
        "n_legs": len(choices),
        "frontier_choices": choices,
        "gave_up": gave_up,
    }


# ---------------------------------------------------------------- driver
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene")
    ap.add_argument("--json", default=None, help="input episodes file (default: repo path)")
    ap.add_argument("--out", default=None, help="output file (default: overwrite input in place)")
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()

    jpath = args.json or f"multigoal_episodes/data/episodes_{args.scene}.json"
    data = json.load(open(jpath))
    episodes = data["episodes"]

    sim = make_sim(args.scene, gpu=args.gpu)
    t0 = time.time()
    found_n = 0
    ratios = []
    for ep in episodes:
        tslot = ep["traces"]["tight"]["target_slot"]
        target = ep[tslot]
        tr = explore(sim, target, ep["start_pose"])
        tr["target_slot"] = tslot
        tr["target_category"] = target["category"]
        ep["traces"]["explore"] = tr
        found_n += tr["found"]
        tight = ep["traces"]["tight"]["steps"]
        ratios.append(tr["steps"] / max(1, tight))
        print(f"  {ep['episode_id']:>22}  {target['category']:<16} "
              f"tight={tight:>3} explore={tr['steps']:>4} found={str(tr['found']):<5} "
              f"legs={tr['n_legs']:>2} ratio={tr['steps']/max(1,tight):>5.2f}x", flush=True)
    sim.close()

    data.setdefault("explore_meta", {})
    data["explore_meta"] = {"cell": CELL, "reveal_r": REVEAL_R, "see_r": SEE_R,
                            "half_fov_deg": 45.0, "scan_every": SCAN_EVERY,
                            "detect": "geometric_visibility"}
    out = args.out or jpath
    with open(out, "w") as f:
        json.dump(data, f, indent=2)
    r = np.array(ratios)
    print(f"\n{args.scene}: found {found_n}/{len(episodes)} ({found_n/len(episodes):.0%}) | "
          f"explore/tight ratio med={np.median(r):.1f}x max={r.max():.1f}x | "
          f"{time.time()-t0:.1f}s -> {out}")


if __name__ == "__main__":
    main()
