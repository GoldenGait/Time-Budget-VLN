"""Render a frontier-exploration episode as [ egocentric RGB | growing occupancy map ]
side-by-side mp4. Plans with the validated smoke core, then replays the primitive
trace with an RGB camera, animating the reveal + trajectory + target.

  python render_frontier_video.py 17DRP5sb8fy --episode 6
"""
import argparse, json, math, os, sys
import numpy as np
import imageio
import habitat_sim
from habitat_sim.utils.common import quat_from_angle_axis
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(__file__))
import smoke_frontier as sf   # Grid, explore, detect, reveal, apos, fwd_xz, constants

FORWARD_M, TURN_DEG = 0.25, 15.0
RGB_H, RGB_W = 480, 640
MAP_SCALE = 22          # px per occupancy cell
STRIDE = 2              # capture every Nth primitive
FPS = 14


def make_rgb_sim(scene, gpu=1):
    glb = f"{sf.MP3D}/{scene}/{scene}.glb"
    b = habitat_sim.SimulatorConfiguration()
    b.scene_id = glb; b.enable_physics = False; b.gpu_device_id = gpu
    s = habitat_sim.CameraSensorSpec()
    s.uuid, s.sensor_type = "rgb", habitat_sim.SensorType.COLOR
    s.resolution = [RGB_H, RGB_W]; s.position = [0.0, 1.25, 0.0]
    ac = habitat_sim.agent.AgentConfiguration()
    ac.sensor_specifications = [s]
    ac.action_space = {
        "move_forward": habitat_sim.agent.ActionSpec("move_forward", habitat_sim.agent.ActuationSpec(amount=FORWARD_M)),
        "turn_left": habitat_sim.agent.ActionSpec("turn_left", habitat_sim.agent.ActuationSpec(amount=TURN_DEG)),
        "turn_right": habitat_sim.agent.ActionSpec("turn_right", habitat_sim.agent.ActuationSpec(amount=TURN_DEG)),
    }
    return habitat_sim.Simulator(habitat_sim.Configuration(b, [ac]))


def grid_base_img(grid):
    img = np.full((grid.nx, grid.nz, 3), 22, np.uint8)
    img[grid.nav] = (70, 70, 80)
    img[grid.seen == sf.FREE] = (140, 216, 158)
    img[grid.seen == sf.OCC] = (200, 92, 76)
    disp = np.flipud(np.transpose(img, (1, 0, 2)))     # (nz,nx,3), +z up
    im = Image.fromarray(disp).resize((grid.nx * MAP_SCALE, grid.nz * MAP_SCALE), Image.NEAREST)
    return im


def cell_px(grid, x, z):
    i, j = grid.wc(x, z)
    col = int((i + 0.5) * MAP_SCALE)
    row = int((grid.nz - 1 - j + 0.5) * MAP_SCALE)
    return col, row


def map_panel(grid, traj_xz, target, start_xz, agent_xz, agent_fwd, status):
    im = grid_base_img(grid); d = ImageDraw.Draw(im)
    if len(traj_xz) >= 2:
        pts = [cell_px(grid, x, z) for x, z in traj_xz]
        d.line(pts, fill=(40, 110, 230), width=3)
    sx, sy = cell_px(grid, *start_xz)
    d.ellipse([sx - 7, sy - 7, sx + 7, sy + 7], fill=(0, 210, 255))
    tx, ty = cell_px(grid, target["navpoint"][0], target["navpoint"][2])
    d.polygon([(tx, ty - 12), (tx + 4, ty - 4), (tx + 12, ty - 4), (tx + 6, ty + 3),
               (tx + 8, ty + 12), (tx, ty + 6), (tx - 8, ty + 12), (tx - 6, ty + 3),
               (tx - 12, ty - 4), (tx - 4, ty - 4)], fill=(255, 205, 0))
    ax, ay = cell_px(grid, *agent_xz)
    hx, hy = ax + int(agent_fwd[0] * 22), ay - int(agent_fwd[1] * 22)
    d.line([(ax, ay), (hx, hy)], fill=(255, 60, 60), width=3)
    d.ellipse([ax - 6, ay - 6, ax + 6, ay + 6], fill=(255, 60, 60))
    im = im.resize((int(im.width * RGB_H / im.height), RGB_H), Image.NEAREST)
    d2 = ImageDraw.Draw(im); d2.rectangle([0, 0, im.width, 22], fill=(0, 0, 0))
    d2.text((6, 6), status, fill=(255, 255, 255))
    return np.asarray(im)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene"); ap.add_argument("--episode", type=int, default=6)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    epj = f"/home/maitree-tiamat/Time-Budget-VLN/multigoal_episodes/data/episodes_{a.scene}.json"
    ep = json.load(open(epj))["episodes"][a.episode]
    target = ep[ep["traces"]["tight"]["target_slot"]]
    cat = target["category"]; start = ep["start_pose"]

    sim = make_rgb_sim(a.scene)
    # 1) plan (moves the agent; we only want the primitive trace)
    prims, found, _, _, _, _, _ = sf.explore(sim, target, start)
    acts = [p for p in prims if p != "stop"]
    print(f"planned: {len(acts)} prims, found={found}; rendering every {STRIDE}...", flush=True)

    # 2) replay from start with the RGB camera, animating the map
    sf.set_agent(sim, start["position"], start["yaw"])
    grid = sf.Grid(sim.pathfinder, start["position"][1])
    grid.reveal(sf.apos(sim), sf.fwd_xz(sim))
    traj = [sf.apos(sim)[[0, 2]].tolist()]
    start_xz = (start["position"][0], start["position"][2])
    frames = []

    def snap(i, n):
        rgb = sim.get_sensor_observations()["rgb"][:, :, :3]
        ax = sf.apos(sim); axz = (float(ax[0]), float(ax[2]))
        mp = map_panel(grid, traj, target, start_xz, axz, sf.fwd_xz(sim),
                       f"find the {cat}  |  step {i}/{n}" + ("  FOUND" if i >= n else ""))
        rp = Image.fromarray(rgb.copy()); dd = ImageDraw.Draw(rp)
        dd.rectangle([0, 0, rp.width, 22], fill=(0, 0, 0)); dd.text((6, 6), "egocentric RGB", fill=(255, 255, 255))
        rp = np.asarray(rp)
        h = min(rp.shape[0], mp.shape[0])
        frames.append(np.concatenate([rp[:h], mp[:h]], axis=1))

    snap(0, len(acts))
    for i, act in enumerate(acts, 1):
        sim.step(act); traj.append(sf.apos(sim)[[0, 2]].tolist())
        grid.reveal(sf.apos(sim), sf.fwd_xz(sim))
        if i % STRIDE == 0 or i == len(acts):
            snap(i, len(acts))
    # hold the final "found" frame
    for _ in range(FPS): frames.append(frames[-1])

    out = a.out or f"/tmp/claude-1013/-home-maitree-tiamat-Time-Budget-VLN/505270d0-b7a6-4273-8f99-9071cbeda526/scratchpad/frontier_{ep['episode_id']}.mp4"
    imageio.mimsave(out, frames, fps=FPS, macro_block_size=2)
    print(f"{len(frames)} frames -> {out}", flush=True)
    sim.close()


if __name__ == "__main__":
    main()
