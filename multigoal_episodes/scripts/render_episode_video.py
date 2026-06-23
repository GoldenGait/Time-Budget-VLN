"""Render a NaVILA-style triptych mp4 of the expert navigating a generated
episode: [ egocentric RGB | depth | top-down map with trajectory + goals ].

mode=both : drive the budget 'both_tour' ordering S -> first -> second
mode=near : drive only to the nearest goal (tight-budget behavior)

Usage:
  python render_episode_video.py GdvgFV5R1Z5 --ep 0 --mode both
"""
import argparse
import json
import math

import numpy as np
import imageio
import habitat_sim
from habitat_sim.utils.common import quat_from_angle_axis, quat_rotate_vector
from habitat.utils.visualizations import maps
from PIL import Image, ImageDraw

FORWARD_M, TURN_DEG = 0.25, 15.0
SUCCESS_DISTANCE, MAX_LEG_STEPS, FPS = 0.2, 500, 6
PANEL_H = 480
MP3D = "/media/maitree-tiamat/Expansion/NaVILA_data/scene_datasets/mp3d"


def make_sim(scene_id):
    glb = f"{MP3D}/{scene_id}/{scene_id}.glb"
    backend = habitat_sim.SimulatorConfiguration()
    backend.scene_id = glb
    backend.enable_physics = False
    specs = []
    for uuid, stype in [("rgb", habitat_sim.SensorType.COLOR),
                        ("depth", habitat_sim.SensorType.DEPTH)]:
        s = habitat_sim.CameraSensorSpec()
        s.uuid, s.sensor_type = uuid, stype
        s.resolution = [480, 640]
        s.position = [0.0, 1.25, 0.0]
        specs.append(s)
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = specs
    agent_cfg.action_space = {
        "move_forward": habitat_sim.agent.ActionSpec("move_forward", habitat_sim.agent.ActuationSpec(amount=FORWARD_M)),
        "turn_left": habitat_sim.agent.ActionSpec("turn_left", habitat_sim.agent.ActuationSpec(amount=TURN_DEG)),
        "turn_right": habitat_sim.agent.ActionSpec("turn_right", habitat_sim.agent.ActuationSpec(amount=TURN_DEG)),
    }
    return habitat_sim.Simulator(habitat_sim.Configuration(backend, [agent_cfg]))


def geo(pf, a, b):
    sp = habitat_sim.ShortestPath()
    sp.requested_start = np.asarray(a, dtype=np.float32)
    sp.requested_end = np.asarray(b, dtype=np.float32)
    return sp.geodesic_distance if pf.find_path(sp) else math.inf


def world_to_px(pf, shape, world):
    """world xyz -> (col, row) pixel on the top-down map."""
    r, c = maps.to_grid(world[2], world[0], shape, pathfinder=pf)
    return (c, r)


def capture(sim):
    obs = sim.get_sensor_observations()
    st = sim.get_agent(0).get_state()
    fwd = quat_rotate_vector(st.rotation, np.array([0.0, 0.0, -1.0]))
    return {"rgb": obs["rgb"][:, :, :3].copy(),
            "depth": obs["depth"].copy(),
            "pos": np.array(st.position), "fwd": fwd}


def drive_leg(sim, follower, goal_np, snaps):
    pf = sim.pathfinder
    steps = 0
    while steps < MAX_LEG_STEPS:
        if geo(pf, sim.get_agent(0).get_state().position, goal_np) <= SUCCESS_DISTANCE:
            return True
        try:
            a = follower.next_action_along(goal_np)
        except habitat_sim.errors.GreedyFollowerError:
            return False
        if a is None:
            return True
        sim.step(a)
        steps += 1
        snaps.append(capture(sim))
    return False


def colorize_depth(d):
    d = np.clip(d, 0.0, 10.0) / 10.0
    g = (255 * (1.0 - d)).astype(np.uint8)
    return np.stack([g, g, g], axis=-1)


def resize_h(img, h=PANEL_H):
    im = Image.fromarray(img)
    w = int(im.width * h / im.height)
    return np.asarray(im.resize((w, h)))


def build_map_frame(pf, base_rgb, shape, start_px, goal_px, traj_px, agent, label):
    img = Image.fromarray(base_rgb.copy())
    d = ImageDraw.Draw(img)
    rad = max(4, shape[0] // 110)
    # trajectory so far
    if len(traj_px) >= 2:
        d.line(traj_px, fill=(30, 90, 220), width=max(2, rad // 2))
    # start (blue) and goals (green=first/near target shown red? use distinct)
    sx, sy = start_px
    d.rectangle([sx - rad, sy - rad, sx + rad, sy + rad], fill=(30, 90, 220))
    for (gx, gy), col in goal_px:
        d.rectangle([gx - rad, gy - rad, gx + rad, gy + rad], fill=col)
    # agent position + heading
    ax, ay = agent["px"]
    hx, hy = agent["head_px"]
    d.line([(ax, ay), (hx, hy)], fill=(0, 200, 255), width=max(2, rad // 2))
    d.ellipse([ax - rad, ay - rad, ax + rad, ay + rad], fill=(0, 200, 255))
    return np.asarray(img)


def banner(panel, text):
    img = Image.fromarray(panel)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, img.width, 24], fill=(0, 0, 0))
    d.text((6, 6), text, fill=(255, 255, 255))
    return np.asarray(img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene")
    ap.add_argument("--ep", type=int, default=0)
    ap.add_argument("--mode", choices=["both", "near"], default="both")
    ap.add_argument("--json", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    j = args.json or f"multigoal_episodes/data/episodes_{args.scene}.json"
    ep = json.load(open(j))["episodes"][args.ep]
    g1, g2, sp = ep["G1"], ep["G2"], ep["start_pose"]

    sim = make_sim(args.scene)
    pf = sim.pathfinder
    follower = habitat_sim.GreedyGeodesicFollower(
        pf, sim.get_agent(0), goal_radius=FORWARD_M,
        forward_key="move_forward", left_key="turn_left", right_key="turn_right")

    st = habitat_sim.AgentState()
    st.position = np.asarray(sp["position"], dtype=np.float32)
    st.rotation = quat_from_angle_axis(sp["yaw"], np.array([0.0, 1.0, 0.0]))
    sim.get_agent(0).set_state(st)
    follower.reset()

    snaps = [capture(sim)]
    if args.mode == "near":
        near = g1 if ep["segment_costs"]["S_G1"] <= ep["segment_costs"]["S_G2"] else g2
        title = f"TIGHT budget -> nearest only: {near['category']}"
        drive_leg(sim, follower, near["navpoint"], snaps)
        goals_world = [(near["navpoint"], (220, 30, 30))]
    else:
        first, second = (g1, g2) if ep["ordering_for_both"] == "G1,G2" else (g2, g1)
        title = f"LOOSE budget -> both: {first['category']} then {second['category']}"
        drive_leg(sim, follower, first["navpoint"], snaps)
        drive_leg(sim, follower, second["navpoint"], snaps)
        goals_world = [(first["navpoint"], (220, 30, 30)),
                       (second["navpoint"], (30, 200, 30))]

    # precompute top-down map (once)
    tdmap = maps.get_topdown_map(pf, sp["position"][1], map_resolution=1024, draw_border=True)
    base_rgb = maps.TOP_DOWN_MAP_COLORS[tdmap]
    shape = tdmap.shape[0:2]
    start_px = world_to_px(pf, shape, sp["position"])
    goal_px = [(world_to_px(pf, shape, w), c) for w, c in goals_world]

    frames = []
    traj_px = []
    for i, s in enumerate(snaps):
        apx = world_to_px(pf, shape, s["pos"])
        hpx = world_to_px(pf, shape, s["pos"] + 0.6 * s["fwd"])
        traj_px.append(apx)
        mp = build_map_frame(pf, base_rgb, shape, start_px, goal_px, traj_px,
                             {"px": apx, "head_px": hpx}, title)
        panels = [resize_h(s["rgb"]), resize_h(colorize_depth(s["depth"])), resize_h(mp)]
        frame = np.concatenate(panels, axis=1)
        frames.append(banner(frame, f"{title} | step {i}/{len(snaps)-1}"))

    out = args.out or f"multigoal_episodes/videos/ep{args.ep}_{args.mode}_{args.scene}.mp4"
    # macro_block_size=2 pads odd width/height up to even (libx264 + yuv420p
    # needs even dims; some scenes' map panel makes the triptych width odd)
    imageio.mimsave(out, frames, fps=FPS, macro_block_size=2)
    print(f"{len(frames)} frames -> {out}")
    sim.close()


if __name__ == "__main__":
    main()
