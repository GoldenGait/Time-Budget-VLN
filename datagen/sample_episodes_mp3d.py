"""Phase 2.1 (MP3D) - sample object-search episodes on MP3D scenes.

MP3D semantics work in habitat (unlike HM3D-Sem v0.2 here): object obb.center positions are
valid, so we bake per-scene object categories+positions into the episode JSON. The expert then
needs only depth (reveal) + these positions (room-voting + geometric detection) -- no runtime
semantic sensor. Targets are the 6 goal-category instances; viewpoints = navmesh point nearest
each target object.

  python sample_episodes_mp3d.py --eps-per-cat 5 --scenes 17DRP5sb8fy --out datagen/ep_mp3d.json
"""
import argparse, glob, json, math, os
import numpy as np
import habitat_sim
import magnum as mn
from habitat_sim.utils.common import quat_from_angle_axis

MP3D = "/data/maitree-tiamat/navila/mp3d/scene_datasets/mp3d"
CATS = ["chair", "bed", "plant", "toilet", "tv_monitor", "sofa"]
SYN = {"couch": "sofa", "tv": "tv_monitor", "monitor": "tv_monitor"}
GEO_MIN, GEO_MAX = 4.0, 20.0
MAX_TRIES = 50
CAM_H, HFOV, W, H = 0.88, 79.0, 640, 480
GLOBAL_SEED = 10
STRUCT = {"wall", "floor", "ceiling", "door", "window", "misc", "objects", "void", "unknown", "", "stairs"}


def norm(name):
    n = name.lower().strip().replace(" ", "_")
    return SYN.get(name.lower().strip(), n)


def make_sim(glb, gpu=0):
    b = habitat_sim.SimulatorConfiguration(); b.scene_id = glb; b.gpu_device_id = gpu; b.enable_physics = False
    s = habitat_sim.CameraSensorSpec(); s.uuid = "depth"; s.sensor_type = habitat_sim.SensorType.DEPTH
    s.resolution = [H, W]; s.position = mn.Vector3(0.0, CAM_H, 0.0); s.hfov = HFOV
    ag = habitat_sim.agent.AgentConfiguration(); ag.sensor_specifications = [s]
    ag.action_space = {"move_forward": habitat_sim.agent.ActionSpec("move_forward", habitat_sim.agent.ActuationSpec(amount=0.25)),
        "turn_left": habitat_sim.agent.ActionSpec("turn_left", habitat_sim.agent.ActuationSpec(amount=30.0)),
        "turn_right": habitat_sim.agent.ActionSpec("turn_right", habitat_sim.agent.ActuationSpec(amount=30.0))}
    sim = habitat_sim.Simulator(habitat_sim.Configuration(b, [ag]))
    ns = habitat_sim.NavMeshSettings(); ns.set_defaults(); ns.agent_radius = 0.18; ns.agent_height = CAM_H
    sim.recompute_navmesh(sim.pathfinder, ns)
    return sim


def geo(pf, a, b):
    sp = habitat_sim.ShortestPath(); sp.requested_start = np.asarray(a, np.float32); sp.requested_end = np.asarray(b, np.float32)
    return sp.geodesic_distance if pf.find_path(sp) else math.inf


def scene_objects(sim, prior):
    """all mapped non-structural objects -> (mp, [x,y,z]); and 6-cat targets -> per cat."""
    objs, targets = [], {c: {"object_positions": [], "viewpoints": []} for c in CATS}
    pf = sim.pathfinder
    for o in sim.semantic_scene.objects:
        if o is None:
            continue
        raw = o.category.name()
        mp = norm(raw)
        c = o.obb.center
        pos = [float(c[0]), float(c[1]), float(c[2])]
        if raw.lower().strip() in STRUCT:
            continue
        if mp in prior:
            objs.append([mp, pos[0], pos[2]])                 # [cat, x, z] for voting
        cat6 = mp if mp in CATS else None
        if cat6:
            npt = pf.snap_point(pos)
            if npt is not None and not np.any(np.isnan(npt)):
                targets[cat6]["object_positions"].append(pos)
                targets[cat6]["viewpoints"].append([float(npt[0]), float(npt[1]), float(npt[2])])
    targets = {c: v for c, v in targets.items() if v["object_positions"]}
    return objs, targets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eps-per-cat", type=int, default=5)
    ap.add_argument("--scenes", nargs="+", default=None)
    ap.add_argument("--prior", default="datagen/prior_table.json")
    ap.add_argument("--out", default="datagen/ep_mp3d.json")
    ap.add_argument("--gpu", type=int, default=0)
    a = ap.parse_args()
    prior = json.load(open(a.prior))["prior"]
    glbs = sorted(glob.glob(os.path.join(MP3D, "*", "*.glb")))
    out = {"meta": {"dataset": "mp3d", "eps_per_cat": a.eps_per_cat, "cats": CATS, "geo_band": [GEO_MIN, GEO_MAX]}, "scenes": {}}
    n_ep = 0
    for gi, glb in enumerate(glbs):
        stem = os.path.basename(glb).replace(".glb", "")
        if a.scenes and stem not in a.scenes:
            continue
        sim = make_sim(glb, a.gpu); pf = sim.pathfinder
        objs, targets = scene_objects(sim, prior)
        lo, hi = pf.get_bounds()
        eps = []
        for cat in sorted(targets):
            vps = np.asarray(targets[cat]["viewpoints"], np.float32)
            rng = np.random.RandomState((GLOBAL_SEED + gi + hash(cat)) % (2**31))
            got = 0
            for _ in range(a.eps_per_cat * MAX_TRIES):
                if got >= a.eps_per_cat:
                    break
                p = np.array([rng.uniform(lo[0], hi[0]), rng.uniform(lo[1], hi[1]), rng.uniform(lo[2], hi[2])], np.float32)
                sn = pf.snap_point(p)
                if sn is None or np.any(np.isnan(sn)):
                    continue
                sf = vps[np.abs(vps[:, 1] - float(sn[1])) < 0.5]          # same floor
                if len(sf) == 0:
                    continue
                dmin = min((geo(pf, sn, v) for v in sf), default=math.inf)
                if not (GEO_MIN <= dmin <= GEO_MAX):
                    continue
                eps.append({"episode_id": f"{cat}_{got}", "category": cat,
                            "start_position": [float(x) for x in sn], "start_yaw": float(rng.uniform(0, 2 * math.pi))})
                got += 1
        out["scenes"][stem] = {"glb": glb, "targets": targets, "objects": objs, "episodes": eps}
        n_ep += len(eps)
        print(f"[{stem}] targets={sorted(targets)} objects={len(objs)} -> {len(eps)} episodes", flush=True)
        sim.close()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(out, open(a.out, "w"))
    print(f"\nwrote {n_ep} episodes across {len(out['scenes'])} scenes -> {a.out}")


if __name__ == "__main__":
    main()
