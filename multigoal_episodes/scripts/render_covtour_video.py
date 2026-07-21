"""Render a COVERAGE-TOUR (option A) episode as [ egocentric RGB | map with viewpoints ]
mp4 — same layout as render_frontier_video.py, for a head-to-head comparison.

  python render_covtour_video.py 17DRP5sb8fy --episode 0
"""
import argparse, json, math, os, sys
import numpy as np
import imageio
import habitat_sim
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(__file__))
import coverage_tour_demo as ct   # Grid, compute_viewpoints, run, set_agent, apos, fwd_xz, constants

RGB_H, RGB_W = 480, 640
MAP_SCALE = 22
STRIDE = 2
FPS = 14


def make_rgb_sim(scene, gpu=1):
    b = habitat_sim.SimulatorConfiguration()
    b.scene_id = f"{ct.MP3D}/{scene}/{scene}.glb"; b.enable_physics = False; b.gpu_device_id = gpu
    s = habitat_sim.CameraSensorSpec()
    s.uuid, s.sensor_type = "rgb", habitat_sim.SensorType.COLOR
    s.resolution = [RGB_H, RGB_W]; s.position = [0.0, 1.25, 0.0]
    ac = habitat_sim.agent.AgentConfiguration(); ac.sensor_specifications = [s]
    ac.action_space = {
        "move_forward": habitat_sim.agent.ActionSpec("move_forward", habitat_sim.agent.ActuationSpec(amount=ct.FORWARD_M)),
        "turn_left": habitat_sim.agent.ActionSpec("turn_left", habitat_sim.agent.ActuationSpec(amount=ct.TURN_DEG)),
        "turn_right": habitat_sim.agent.ActionSpec("turn_right", habitat_sim.agent.ActuationSpec(amount=ct.TURN_DEG)),
    }
    return habitat_sim.Simulator(habitat_sim.Configuration(b, [ac]))


def cell_px(grid, x, z):
    i, j = grid.wc(x, z)
    return int((i + .5) * MAP_SCALE), int((grid.nz - 1 - j + .5) * MAP_SCALE)


def base_img(grid):
    img = np.full((grid.nx, grid.nz, 3), 22, np.uint8)
    img[grid.nav] = (70, 70, 80)
    disp = np.flipud(np.transpose(img, (1, 0, 2)))
    return Image.fromarray(disp).resize((grid.nx * MAP_SCALE, grid.nz * MAP_SCALE), Image.NEAREST)


def map_panel(grid, vps, order, traj, target, start_xz, agent_xz, agent_fwd, status):
    im = base_img(grid); d = ImageDraw.Draw(im)
    # tour order (faint dashed-ish) + viewpoints
    ow = [cell_px(grid, *grid.cw(*vp[0])) for vp in order]
    if len(ow) >= 2:
        d.line(ow, fill=(150, 110, 40), width=2)
    for (i, j) in vps:
        x, y = cell_px(grid, *grid.cw(i, j))
        d.rectangle([x - 5, y - 5, x + 5, y + 5], fill=(255, 170, 0))
    if len(traj) >= 2:
        d.line([cell_px(grid, x, z) for x, z in traj], fill=(40, 110, 230), width=3)
    sx, sy = cell_px(grid, *start_xz); d.ellipse([sx - 7, sy - 7, sx + 7, sy + 7], fill=(0, 210, 255))
    tx, ty = cell_px(grid, target["navpoint"][0], target["navpoint"][2])
    d.polygon([(tx, ty - 12), (tx + 4, ty - 4), (tx + 12, ty - 4), (tx + 6, ty + 3), (tx + 8, ty + 12),
               (tx, ty + 6), (tx - 8, ty + 12), (tx - 6, ty + 3), (tx - 12, ty - 4), (tx - 4, ty - 4)], fill=(255, 205, 0))
    ax, ay = cell_px(grid, *agent_xz); hx, hy = ax + int(agent_fwd[0] * 22), ay - int(agent_fwd[1] * 22)
    d.line([(ax, ay), (hx, hy)], fill=(255, 60, 60), width=3); d.ellipse([ax - 6, ay - 6, ax + 6, ay + 6], fill=(255, 60, 60))
    im = im.resize((int(im.width * RGB_H / im.height), RGB_H), Image.NEAREST)
    dd = ImageDraw.Draw(im); dd.rectangle([0, 0, im.width, 22], fill=(0, 0, 0)); dd.text((6, 6), status, fill=(255, 255, 255))
    return np.asarray(im)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("scene"); ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    ep = json.load(open(f"/home/maitree-tiamat/Time-Budget-VLN/multigoal_episodes/data/episodes_{a.scene}.json"))["episodes"][a.episode]
    target = ep[ep["traces"]["tight"]["target_slot"]]; cat = target["category"]; start = ep["start_pose"]

    sim = make_rgb_sim(a.scene)
    grid = ct.Grid(sim.pathfinder, start["position"][1])
    vps, _ = ct.compute_viewpoints(grid)
    prims, found, _, order = ct.run(sim, target, start, {"grid": grid, "vps": vps})
    acts = [p for p in prims if p != "stop"]
    print(f"covtour: {len(vps)} viewpoints, {len(acts)} prims, found={found}; rendering...", flush=True)

    ct.set_agent(sim, start["position"], start["yaw"])
    traj = [ct.apos(sim)[[0, 2]].tolist()]; start_xz = (start["position"][0], start["position"][2])
    frames = []

    def snap(i, n):
        rgb = sim.get_sensor_observations()["rgb"][:, :, :3]
        ap = ct.apos(sim); axz = (float(ap[0]), float(ap[2]))
        mp = map_panel(grid, vps, order, traj, target, start_xz, axz, ct.fwd_xz(sim),
                       f"find the {cat}  |  step {i}/{n}" + ("  FOUND" if i >= n else ""))
        rp = Image.fromarray(rgb.copy()); dd = ImageDraw.Draw(rp)
        dd.rectangle([0, 0, rp.width, 22], fill=(0, 0, 0)); dd.text((6, 6), "egocentric RGB (COVERAGE TOUR)", fill=(255, 255, 255))
        rp = np.asarray(rp); h = min(rp.shape[0], mp.shape[0])
        frames.append(np.concatenate([rp[:h], mp[:h]], axis=1))

    snap(0, len(acts))
    for i, act in enumerate(acts, 1):
        sim.step(act); traj.append(ct.apos(sim)[[0, 2]].tolist())
        if i % STRIDE == 0 or i == len(acts):
            snap(i, len(acts))
    for _ in range(FPS): frames.append(frames[-1])

    out = a.out or f"/tmp/claude-1013/-home-maitree-tiamat-Time-Budget-VLN/505270d0-b7a6-4273-8f99-9071cbeda526/scratchpad/covtour_{ep['episode_id']}.mp4"
    imageio.mimsave(out, frames, fps=FPS, macro_block_size=2)
    print(f"{len(frames)} frames -> {out}", flush=True)
    sim.close()


if __name__ == "__main__":
    main()
