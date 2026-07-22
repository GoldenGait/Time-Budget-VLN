"""Phase 2.1 - sample object-search episodes on HM3D-Sem train scenes.

Detection targets are the ObjectNav goal instances (object_id + viewpoints), so the
expert's stop and the ObjectNav success metric use the identical goal definition.
For each (scene, category present) we sample starts that are:
  - navigable, geodesic in [GEO_MIN, GEO_MAX] m to the nearest target VIEWPOINT,
  - NOT visible at start (no target instance rendered >= START_VIS_PX semantic pixels).
Deterministic: per-(scene,category) seed, np.random sampling, sorted iteration.

Output: episodes_<split>.json = {meta, scenes:{stem:{scene_id, targets, episodes}}}.

  python sample_episodes.py --split train --eps-per-cat 15 --out datagen/episodes_train.json
"""
import argparse, glob, gzip, json, math, os
import numpy as np
import habitat_sim
import magnum as mn
from habitat_sim.utils.common import quat_from_angle_axis

HM3D = "/data/maitree-tiamat/navila/scene_datasets/versioned_data/hm3d-0.2/hm3d"
CFG = f"{HM3D}/hm3d_annotated_basis.scene_dataset_config.json"
ONAV = "/data/maitree-tiamat/objectnav_data/objectnav_hm3d_v2/{split}/content"
CATS = ["chair", "bed", "plant", "toilet", "tv_monitor", "sofa"]

GEO_MIN, GEO_MAX = 4.0, 20.0
START_VIS_PX = 50          # target counts as "visible" at >= this many semantic pixels
MAX_TRIES = 50
CAM_H, HFOV, W, H = 0.88, 79.0, 640, 480
GLOBAL_SEED = 10


def make_sim(scene_id, gpu=0):
    b = habitat_sim.SimulatorConfiguration()
    b.scene_dataset_config_file = CFG
    b.scene_id = scene_id
    b.gpu_device_id = gpu
    b.enable_physics = False
    s = habitat_sim.CameraSensorSpec()
    s.uuid = "semantic"; s.sensor_type = habitat_sim.SensorType.SEMANTIC
    s.resolution = [H, W]; s.position = mn.Vector3(0.0, CAM_H, 0.0); s.hfov = HFOV
    ag = habitat_sim.agent.AgentConfiguration()
    ag.sensor_specifications = [s]
    ag.action_space = {
        "move_forward": habitat_sim.agent.ActionSpec("move_forward", habitat_sim.agent.ActuationSpec(amount=0.25)),
        "turn_left": habitat_sim.agent.ActionSpec("turn_left", habitat_sim.agent.ActuationSpec(amount=30.0)),
        "turn_right": habitat_sim.agent.ActionSpec("turn_right", habitat_sim.agent.ActuationSpec(amount=30.0)),
    }
    sim = habitat_sim.Simulator(habitat_sim.Configuration(b, [ag]))
    ns = habitat_sim.NavMeshSettings(); ns.set_defaults()
    ns.agent_radius = 0.18; ns.agent_height = CAM_H
    sim.recompute_navmesh(sim.pathfinder, ns)
    return sim


def geodesic(pf, a, b):
    sp = habitat_sim.ShortestPath()
    sp.requested_start = np.asarray(a, np.float32); sp.requested_end = np.asarray(b, np.float32)
    return sp.geodesic_distance if pf.find_path(sp) else math.inf


def set_pose(sim, pos, yaw):
    st = sim.get_agent(0).get_state()
    st.position = np.asarray(pos, np.float32)
    st.rotation = quat_from_angle_axis(float(yaw), np.array([0.0, 1.0, 0.0]))
    sim.get_agent(0).set_state(st)


def target_pixels(sim, target_ids):
    sem = sim.get_sensor_observations()["semantic"]
    return {tid: int((sem == tid).sum()) for tid in target_ids}


def scene_targets(sim, gbc, sk):
    """Per-category: object_ids, object positions, and viewpoint positions (from ObjectNav goals)."""
    out = {}
    for cat in CATS:
        goals = gbc.get(f"{sk}_{cat}", [])
        if not goals:
            continue
        ids, opos, vps = [], [], []
        for g in goals:
            ids.append(int(g["object_id"]))
            opos.append([float(x) for x in g["position"]])
            for vp in g.get("view_points", []):
                vps.append([float(x) for x in vp["agent_state"]["position"]])
        out[cat] = {"object_ids": ids, "object_positions": opos, "viewpoints": vps}
    return out


def sample_scene(sim, targets, eps_per_cat, base_seed):
    pf = sim.pathfinder
    lo, hi = pf.get_bounds()
    episodes = []
    for cat in sorted(targets):
        tgt = targets[cat]
        vps = np.asarray(tgt["viewpoints"], np.float32)
        ids = tgt["object_ids"]
        rng = np.random.RandomState((base_seed + hash(cat)) % (2**31))
        got = 0
        for tries in range(eps_per_cat * MAX_TRIES):
            if got >= eps_per_cat:
                break
            p = np.array([rng.uniform(lo[0], hi[0]), rng.uniform(lo[1], hi[1]), rng.uniform(lo[2], hi[2])], np.float32)
            snapped = pf.snap_point(p)
            if snapped is None or np.any(np.isnan(snapped)):
                continue
            # SAME-FLOOR only: the occupancy grid is a 2D slice at the start height, so the
            # target must be on the agent's floor (|dy| < 0.5). Cross-floor search needs a
            # multi-level map (out of scope for v1).
            sf = vps[np.abs(vps[:, 1] - float(snapped[1])) < 0.5]
            if len(sf) == 0:
                continue
            dmin = min((geodesic(pf, snapped, v) for v in sf), default=math.inf)
            if not (GEO_MIN <= dmin <= GEO_MAX):
                continue
            yaw = float(rng.uniform(0, 2 * math.pi))
            set_pose(sim, snapped, yaw)
            if max(target_pixels(sim, ids).values(), default=0) >= START_VIS_PX:
                continue  # target already visible -> reject
            episodes.append({
                "episode_id": f"{cat}_{got}",
                "category": cat,
                "start_position": [float(x) for x in snapped],
                "start_yaw": yaw,
                "geo_to_target": float(dmin),
                "seed": int(base_seed + hash(cat)) % (2**31),
            })
            got += 1
    return episodes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--eps-per-cat", type=int, default=15)
    ap.add_argument("--scenes", nargs="+", default=None, help="limit to these scene stems (smoke)")
    ap.add_argument("--out", default="datagen/episodes_train.json")
    ap.add_argument("--gpu", type=int, default=0)
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(ONAV.format(split=a.split), "*.json.gz")))
    out = {"meta": {"split": a.split, "eps_per_cat": a.eps_per_cat, "cats": CATS,
                    "geo_band": [GEO_MIN, GEO_MAX], "start_vis_px": START_VIS_PX}, "scenes": {}}
    n_ep = 0
    for fi, f in enumerate(files):
        d = json.load(gzip.open(f))
        eps0 = d["episodes"]
        if not eps0:
            continue
        scene_id = eps0[0]["scene_id"]
        sk = scene_id.split("/")[-1]                     # e.g. 1S7LAXRdDqK.basis.glb
        stem = sk.replace(".basis.glb", "")
        if a.scenes and stem not in a.scenes:
            continue
        sim = make_sim(stem, a.gpu)          # config resolves by stem; ObjectNav scene_id has an unrecognized hm3d_v0.2/ prefix
        targets = scene_targets(sim, d["goals_by_category"], sk)
        eps = sample_scene(sim, targets, a.eps_per_cat, GLOBAL_SEED + fi)
        # strip pixel-heavy viewpoints down for storage? keep for detection alignment.
        out["scenes"][stem] = {"scene_id": scene_id, "targets": targets, "episodes": eps}
        n_ep += len(eps)
        print(f"[{stem}] cats={sorted(targets)} -> {len(eps)} episodes "
              f"({', '.join(sorted(set(e['category'] for e in eps)))})", flush=True)
        sim.close()

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(out, open(a.out, "w"))
    print(f"\nwrote {n_ep} episodes across {len(out['scenes'])} scenes -> {a.out}")


if __name__ == "__main__":
    main()
