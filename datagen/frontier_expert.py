"""Phase 2.2-2.6 - semantic frontier expert (Layer 1) for budget-conditioned object search.

Privileged (allowed): navmesh, semantic annotations, target instance positions (STOP check ONLY).
Leak-safe: the PATH never reads the target position; frontier choice uses only revealed
non-target objects (voting) + info-gain + geodesic. Target enters only at detection.

Per episode (from sample_episodes.py) produce:
  loose trace  = frontier search until target seen+faced -> stop
  tight trace  = loose truncated at end of frontier leg 1 -> stop
Deterministic replay: primitive list + start pose (GreedyGeodesicFollower is deterministic).

  python frontier_expert.py --episodes datagen/episodes_train.json --out datagen/traces_train.json
"""
import argparse, collections, json, math, os
import numpy as np
import habitat_sim
import magnum as mn
from habitat_sim.utils.common import quat_from_angle_axis

FORWARD_M, TURN_DEG = 0.25, 30.0
CELL = 0.25
REVEAL_R = 3.0                 # simple omni reveal radius (navmesh + LoS)
IG_R = 3.0                     # info-gain radius around a frontier centroid
VOTE_R = 3.0                   # observed-object voting radius around a frontier centroid
SEE_DIST = 1.0                 # geodesic to a target viewpoint to be "at" the goal
HALF_FOV = math.radians(39.5)  # forward half-HFOV (79/2)
FACE_TOL = math.radians(TURN_DEG / 2.0)   # center target within half a turn-step
FACE_CAP = 3                   # turn-to-center cap (primitives; counted in T)
DETECT_PX = 50                 # target must render >= this many semantic px to confirm
OBJ_MIN_PX = 30                # non-target object counts as "observed" at >= this many semantic px
MIN_CLUSTER = 2                # min frontier cluster size (0.5m); small so doorway frontiers survive
LEG_CAP = 300
SAFETY = 2000
TURNS_360 = int(round(360.0 / TURN_DEG))   # 12 turns at 30deg = full look-around
SCAN_EVERY = 12                            # mid-leg 360deg scan every N forward steps (~3m)
W_IG, W_SEM, W_GEO = 1.0, 2.0, 0.1
EPS = 0.02
CAM_H, HFOV, RW, RH = 0.88, 79.0, 640, 480
UNKNOWN, FREE, OCC = -1, 0, 1

HM3D = "/data/maitree-tiamat/navila/scene_datasets/versioned_data/hm3d-0.2/hm3d"
CFG = f"{HM3D}/hm3d_annotated_basis.scene_dataset_config.json"

# HM3D raw category -> mpcat40 (prior_table key). Unmapped -> no vote.
SYN = {"couch": "sofa", "potted plant": "plant", "houseplant": "plant", "tv": "tv_monitor",
       "television": "tv_monitor", "monitor": "tv_monitor", "tv monitor": "tv_monitor",
       "armchair": "chair", "office chair": "chair", "kitchen cabinet": "cabinet",
       "sink cabinet": "cabinet", "coffee table": "table", "side table": "table",
       "nightstand": "chest_of_drawers", "dresser": "chest_of_drawers"}


def normalize_cat(name, prior):
    n = name.lower().strip()
    u = n.replace(" ", "_")
    if u in prior:
        return u
    if n in SYN:
        return SYN[n]
    if u in SYN:
        return SYN[u]
    return None


# ---------------------------------------------------------------- sim
def make_sim(stem, gpu=0):
    b = habitat_sim.SimulatorConfiguration()
    b.scene_dataset_config_file = CFG; b.scene_id = stem
    b.gpu_device_id = gpu; b.enable_physics = False
    s = habitat_sim.CameraSensorSpec()
    s.uuid = "depth"; s.sensor_type = habitat_sim.SensorType.DEPTH
    s.resolution = [RH, RW]; s.position = mn.Vector3(0.0, CAM_H, 0.0); s.hfov = HFOV
    ag = habitat_sim.agent.AgentConfiguration(); ag.sensor_specifications = [s]
    ag.action_space = {
        "move_forward": habitat_sim.agent.ActionSpec("move_forward", habitat_sim.agent.ActuationSpec(amount=FORWARD_M)),
        "turn_left": habitat_sim.agent.ActionSpec("turn_left", habitat_sim.agent.ActuationSpec(amount=TURN_DEG)),
        "turn_right": habitat_sim.agent.ActionSpec("turn_right", habitat_sim.agent.ActuationSpec(amount=TURN_DEG)),
    }
    sim = habitat_sim.Simulator(habitat_sim.Configuration(b, [ag]))
    ns = habitat_sim.NavMeshSettings(); ns.set_defaults(); ns.agent_radius = 0.18; ns.agent_height = CAM_H
    sim.recompute_navmesh(sim.pathfinder, ns)
    return sim


def apos(sim):
    return np.asarray(sim.get_agent(0).get_state().position, np.float32)


def fwd_xz(sim):
    q = sim.get_agent(0).get_state().rotation
    th = 2.0 * math.atan2(q.y, q.w)
    return -math.sin(th), -math.cos(th)


def set_pose(sim, pos, yaw):
    st = sim.get_agent(0).get_state()
    st.position = np.asarray(pos, np.float32)
    st.rotation = quat_from_angle_axis(float(yaw), np.array([0.0, 1.0, 0.0]))
    sim.get_agent(0).set_state(st)


def geo(pf, a, b):
    sp = habitat_sim.ShortestPath(); sp.requested_start = np.asarray(a, np.float32); sp.requested_end = np.asarray(b, np.float32)
    return sp.geodesic_distance if pf.find_path(sp) else math.inf


def los(pf, a, b, fy):
    n = max(2, int(math.hypot(b[0] - a[0], b[2] - a[2]) / 0.15))
    return all(pf.is_navigable([a[0] + (b[0] - a[0]) * t, fy, a[2] + (b[2] - a[2]) * t]) for t in np.linspace(0, 1, n))


# ---------------------------------------------------------------- occupancy
class Grid:
    def __init__(self, pf, fy):
        lo, _ = pf.get_bounds()
        self.x0, self.z0, self.fy = float(lo[0]), float(lo[2]), float(fy)
        self.nav = np.ascontiguousarray(pf.get_topdown_view(CELL, self.fy).T)
        self.nx, self.nz = self.nav.shape
        self.state = np.full((self.nx, self.nz), UNKNOWN, np.int8)
        self.nav_cells = np.argwhere(self.nav)          # [N,2] navigable cell coords (for fast reveal)

    def cw(self, i, j):
        return self.x0 + (i + 0.5) * CELL, self.z0 + (j + 0.5) * CELL

    def wc(self, x, z):
        return int((x - self.x0) / CELL), int((z - self.z0) / CELL)

    def _los_grid(self, ci, cj, i, j):
        """Grid-space line of sight: every cell between (ci,cj) and (i,j) must be navigable.
        Pure array lookups (no pathfinder calls) -> fast, and gives real wall occlusion so the
        reveal does NOT bleed through walls into unvisited rooms."""
        n = max(abs(i - ci), abs(j - cj))
        if n == 0:
            return True
        for s in range(1, n):
            gi = ci + (i - ci) * s // n
            gj = cj + (j - cj) * s // n
            if not self.nav[gi, gj]:
                return False
        return True

    def reveal_depth(self, depth, ax, az, fx, fz):
        """Depth-projection reveal (fills the visible region, not just ray lines): for each
        navigable cell in range, project it into the camera; if it is inside the FOV and closer
        than the nearest surface in its column (min depth over a body band), it is visible -> FREE.
        Real occlusion via depth, FOV-gated (360deg comes from the look-around scans)."""
        H, W = depth.shape
        cx = (W - 1) / 2.0
        f = (W / 2.0) / math.tan(math.radians(HFOV) / 2.0)
        rx, rz = fz, -fx                                     # right vector (perp to forward)
        band = depth[int(H * 0.35):int(H * 0.80)]            # body-height band -> nearest wall/furniture
        colmin = np.where(band > 0.1, band, np.inf).min(axis=0)   # nearest surface per column
        ci, cj = self.wc(ax, az); r = int(REVEAL_R / CELL)
        d2 = (self.nav_cells[:, 0] - ci) ** 2 + (self.nav_cells[:, 1] - cj) ** 2
        for i, j in self.nav_cells[d2 <= r * r]:
            if self.state[i, j] != UNKNOWN:
                continue
            wx, wz = self.cw(int(i), int(j))
            dx, dz = wx - ax, wz - az
            fwd = dx * fx + dz * fz                           # perpendicular (forward) distance
            if fwd < 0.15:
                self.state[i, j] = FREE                       # basically at the agent
                continue
            u = cx + f * ((dx * rx + dz * rz) / fwd)          # image column of the cell
            if u < 0 or u >= W:
                continue                                      # outside horizontal FOV
            if fwd <= colmin[int(u)] + 0.30:                  # in front of the surface -> visible
                self.state[i, j] = FREE

    def frontiers(self):
        free = np.argwhere(self.state == FREE)
        fr = []
        for i, j in free:
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
                a, b = i + di, j + dj
                if 0 <= a < self.nx and 0 <= b < self.nz and self.state[a, b] == UNKNOWN:
                    fr.append((int(i), int(j))); break
        return fr

    def cluster(self, cells):
        s = set(cells); out = []
        while s:
            seed = s.pop(); comp = [seed]; stack = [seed]
            while stack:
                i, j = stack.pop()
                for di in (-1, 0, 1):
                    for dj in (-1, 0, 1):
                        c = (i + di, j + dj)
                        if c in s:
                            s.discard(c); comp.append(c); stack.append(c)
            out.append(comp)
        return out

    def ig(self, ci, cj):
        r = int(IG_R / CELL); n = 0
        for di in range(-r, r + 1):
            for dj in range(-r, r + 1):
                i, j = ci + di, cj + dj
                if 0 <= i < self.nx and 0 <= j < self.nz and self.state[i, j] == UNKNOWN:
                    n += 1
        return n


# ---------------------------------------------------------------- expert
def explore(sim, ep, targets, prior, objmap):
    pf = sim.pathfinder
    cat = ep["category"]
    tgt = targets[cat]
    tgt_ids = set(tgt["object_ids"])
    tgt_vps = np.asarray(tgt["viewpoints"], np.float32)
    tgt_obj = np.asarray(tgt["object_positions"], np.float32)
    follower = habitat_sim.GreedyGeodesicFollower(pf, sim.get_agent(0), goal_radius=0.5,
                                                  forward_key="move_forward", left_key="turn_left", right_key="turn_right")
    set_pose(sim, ep["start_position"], ep["start_yaw"]); follower.reset()
    grid = Grid(pf, ep["start_position"][1])
    observed = {}     # instance_id -> (mpcat40, [x,z] agent pos at first sighting)  non-target only

    def do_reveal():
        depth = np.asarray(sim.get_sensor_observations()["depth"])
        p = apos(sim); fx, fz = fwd_xz(sim)
        grid.reveal_depth(depth, float(p[0]), float(p[2]), fx, fz)

    def observe():
        # DISABLED: the HM3D-Sem v0.2 semantic sensor renders all-zeros in this habitat-sim
        # build, so runtime object observation is impossible. Room-voting (P_sem) is therefore
        # inert for now (frontier score falls back to info-gain + geodesic). It will be restored
        # via an offline .semantic.glb mesh parse that yields object positions + categories.
        return

    def near_viewpoint():
        p = apos(sim)
        if len(tgt_vps) == 0:
            return math.inf
        de = np.hypot(tgt_vps[:, 0] - p[0], tgt_vps[:, 2] - p[2])   # cheap vectorized euclidean
        close = tgt_vps[de < 2.0]                                    # only these can be within 1m geodesic
        if len(close) == 0:
            return math.inf
        return min(geo(pf, p, v) for v in close)

    def target_px():
        sem = sim.get_sensor_observations()["semantic"]
        return max((int((sem == t).sum()) for t in tgt_ids), default=0)

    def face_target():
        """Turn <=FACE_CAP to center the nearest target object; append turns."""
        turns = []
        p = apos(sim); obj = min(tgt["object_positions"], key=lambda o: (o[0] - p[0]) ** 2 + (o[2] - p[2]) ** 2)
        for _ in range(FACE_CAP):
            p = apos(sim); vx, vz = obj[0] - p[0], obj[2] - p[2]; n = math.hypot(vx, vz)
            if n < 1e-6:
                break
            fx, fz = fwd_xz(sim)
            if (vx * fx + vz * fz) / n >= math.cos(FACE_TOL):
                break
            act = "turn_right" if (fx * vz - fz * vx) > 0 else "turn_left"
            sim.step(act); turns.append(act)
        return turns

    def detect():
        """Geometric (HM3D semantic sensor renders all-zeros, so no pixel check): within 1m
        geodesic of a target viewpoint AND facing the nearest target object (within +-HALF_FOV).
        Viewpoints are defined to see the object, so LoS is implied."""
        if near_viewpoint() >= SEE_DIST:
            return False
        p = apos(sim)
        obj = min(tgt_obj, key=lambda o: (o[0] - p[0]) ** 2 + (o[2] - p[2]) ** 2)
        vx, vz = obj[0] - p[0], obj[2] - p[2]; n = math.hypot(vx, vz)
        if n < 1e-6:
            return True
        fx, fz = fwd_xz(sim)
        return (vx * fx + vz * fz) / n >= math.cos(HALF_FOV)

    def look_around():
        """Turn a full circle one step at a time, revealing + checking detection at each
        heading; stop the instant the target is seen (leaves the agent facing it).
        Appends the scan turns to the trace (counted in the budget)."""
        for _ in range(TURNS_360):
            observe()
            if detect():
                return True
            sim.step("turn_left"); prims.append("turn_left"); do_reveal()
        observe()
        return detect()

    def score_frontier(comp):
        ci = int(round(np.mean([c[0] for c in comp]))); cj = int(round(np.mean([c[1] for c in comp])))
        wx, wz = grid.cw(ci, cj); npt = pf.snap_point([wx, grid.fy, wz])
        if npt is None or np.any(np.isnan(npt)):
            return None
        # room vote from observed non-target objects near the centroid
        votes = collections.Counter()
        for mp, opos in observed.values():          # opos = [x, z]
            if math.hypot(opos[0] - wx, opos[1] - wz) <= VOTE_R and mp in prior:
                for room, pr in prior[mp].items():
                    votes[room] += pr
        room = votes.most_common(1)[0][0] if votes else None
        p_sem = prior.get(cat, {}).get(room, EPS) if room else EPS
        ig = grid.ig(ci, cj); d = geo(pf, apos(sim), npt)
        if not math.isfinite(d):
            return None
        sc = W_IG * math.log(ig + 1) + W_SEM * math.log(p_sem + EPS) - W_GEO * d
        return {"score": sc, "navpoint": [float(x) for x in npt], "room": room, "p_sem": p_sem,
                "ig": ig, "d_geo": float(d), "centroid_xz": [wx, wz]}

    prims, frontier_log, leg_ends = [], [], []
    found, giveup = False, None
    do_reveal()
    if look_around():                        # initial scan from the start pose
        prims += face_target(); found = True

    stuck_pts = []      # frontier navpoints that yielded zero translation (already-at / unreachable)
    while not found:
        if len(prims) >= SAFETY:
            giveup = "safety"; break
        clusters = [c for c in grid.cluster(grid.frontiers()) if len(c) >= MIN_CLUSTER]
        if not clusters:
            giveup = "exhausted"; break
        scored = []
        for k, comp in enumerate(clusters):
            info = score_frontier(comp)
            if not info:
                continue
            np_ = info["navpoint"]
            if any(math.hypot(np_[0] - b[0], np_[2] - b[2]) < 1.0 for b in stuck_pts):
                continue                     # skip blacklisted frontiers
            info["cluster_id"] = k; scored.append(info)
        if not scored:
            giveup = "no_reachable_frontier"; break
        scored.sort(key=lambda s: (-s["score"], s["cluster_id"]))
        f = scored[0]; frontier_log.append({k: f[k] for k in ("score", "room", "p_sem", "ig", "d_geo", "cluster_id")})
        p0 = apos(sim).copy()
        follower.reset(); leg = 0; goal = np.asarray(f["navpoint"], np.float32)
        while leg < LEG_CAP and len(prims) < SAFETY:
            try:
                a = follower.next_action_along(goal)
            except habitat_sim.errors.GreedyFollowerError:
                break
            if a is None:
                break
            sim.step(a); prims.append(a); leg += 1
            do_reveal()
            if detect():
                prims += face_target(); found = True; break
            if a == "move_forward" and leg % SCAN_EVERY == 0 and look_around():   # mid-leg scan
                prims += face_target(); found = True; break
        do_reveal()
        if not found and look_around():      # arrival scan at the reached frontier
            prims += face_target(); found = True
        leg_ends.append(len(prims))
        if found:
            break
        if math.hypot(apos(sim)[0] - p0[0], apos(sim)[2] - p0[2]) < 0.1:   # no translation -> blacklist
            stuck_pts.append(f["navpoint"])

    return {"category": cat, "start_position": ep["start_position"], "start_yaw": ep["start_yaw"],
            "actions": prims, "steps": len(prims), "found": bool(found), "giveup": giveup,
            "n_observed": len(observed), "n_scene_objs": len(objmap),
            "leg_ends": leg_ends, "T_loose": len(prims),
            "T_tight": leg_ends[0] if leg_ends else len(prims),
            "frontier_log": frontier_log}


def scene_object_map(sim, prior):
    """instance_id -> (mpcat40_or_None, raw_category). HM3D semantic objects carry no
    valid obb/aabb geometry here, so positions come from the sensor at sighting time."""
    m = {}
    for o in sim.semantic_scene.objects:
        if o is None:
            continue
        try:
            c = o.category.name()
        except Exception:
            continue
        m[int(o.semantic_id)] = (normalize_cat(c, prior), c)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", default="datagen/episodes_train.json")
    ap.add_argument("--prior", default="datagen/prior_table.json")
    ap.add_argument("--out", default="datagen/traces_train.json")
    ap.add_argument("--scenes", nargs="+", default=None)
    ap.add_argument("--max-eps", type=int, default=0, help="cap episodes per scene (smoke)")
    ap.add_argument("--gpu", type=int, default=0)
    a = ap.parse_args()

    prior = json.load(open(a.prior))["prior"]
    data = json.load(open(a.episodes))
    out = {"meta": {"weights": [W_IG, W_SEM, W_GEO], "reveal_r": REVEAL_R, "see_dist": SEE_DIST,
                    "detect_px": DETECT_PX, "face_cap": FACE_CAP}, "scenes": {}}
    tot = found = 0
    for stem, sc in sorted(data["scenes"].items()):
        if a.scenes and stem not in a.scenes:
            continue
        eps = sc["episodes"][: a.max_eps] if a.max_eps else sc["episodes"]
        if not eps:
            continue
        sim = make_sim(stem, a.gpu)
        objmap = scene_object_map(sim, prior)
        traces = []
        for ep in eps:
            import time as _t; t0 = _t.time()
            tr = explore(sim, ep, sc["targets"], prior, objmap)
            tr["episode_id"] = ep["episode_id"]; traces.append(tr)
            tot += 1; found += tr["found"]
            print(f"    {stem} {ep['episode_id']:12s} found={tr['found']!s:5} T={tr['T_loose']:4d} "
                  f"legs={len(tr['leg_ends'])} {_t.time()-t0:.1f}s", flush=True)
        out["scenes"][stem] = {"traces": traces}
        fr = sum(t["found"] for t in traces)
        print(f"[{stem}] {len(traces)} eps | found {fr}/{len(traces)} "
              f"| T_loose med={int(np.median([t['T_loose'] for t in traces]))} "
              f"T_tight med={int(np.median([t['T_tight'] for t in traces]))}", flush=True)
        sim.close()

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(out, open(a.out, "w"))
    print(f"\nTOTAL found {found}/{tot} ({found/max(1,tot):.0%}) -> {a.out}")


if __name__ == "__main__":
    main()
