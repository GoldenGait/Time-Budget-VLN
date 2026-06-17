"""Read-only recon: render a top-down navmesh map of an MP3D scene with
candidate goal objects marked. Saves a PNG (opens natively in VS Code)."""
import os
import imageio
import habitat_sim
from habitat.utils.visualizations import maps

import sys
MP3D = "/media/maitree-tiamat/Expansion/NaVILA_data/scene_datasets/mp3d"
SCENE_ID = sys.argv[1] if len(sys.argv) > 1 else "17DRP5sb8fy"
SCENE = f"{MP3D}/{SCENE_ID}/{SCENE_ID}.glb"
OUT = "/home/maitree-tiamat/Time-Budget-VLN/multigoal_episodes/maps"

# Goal-worthy mpcat40 categories (the whitelist we reconned)
WHITELIST = {"bed", "sofa", "toilet", "sink", "shower", "tv_monitor",
             "table", "counter", "cabinet", "chest_of_drawers", "stool", "chair"}


def make_sim(scene):
    backend = habitat_sim.SimulatorConfiguration()
    backend.scene_id = scene
    backend.enable_physics = False
    sensor = habitat_sim.CameraSensorSpec()
    sensor.uuid = "rgb"
    sensor.sensor_type = habitat_sim.SensorType.COLOR
    sensor.resolution = [512, 512]
    sensor.position = [0.0, 1.5, 0.0]
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [sensor]
    return habitat_sim.Simulator(habitat_sim.Configuration(backend, [agent_cfg]))


def main():
    sim = make_sim(SCENE)
    pf = sim.pathfinder
    assert pf.is_loaded, "navmesh not loaded"

    # floor height = y of a random navigable point
    start = pf.get_random_navigable_point()
    height = start[1]
    print(f"navmesh bounds: {pf.get_bounds()}  floor height ~ {height:.2f}")

    # top-down occupancy map (no GPU needed)
    tdmap = maps.get_topdown_map(pf, height, map_resolution=1024, draw_border=True)
    rgb = maps.TOP_DOWN_MAP_COLORS[tdmap]

    # overlay goal-candidate objects from semantic annotations
    objs = sim.semantic_scene.objects
    n_marked = 0
    by_cat = {}
    for o in objs:
        if o is None or o.category is None:
            continue
        cat = o.category.name()
        if cat not in WHITELIST:
            continue
        c = o.aabb.center  # world xyz
        if abs(c[1] - height) > 1.5:   # keep this floor only
            continue
        gx, gy = maps.to_grid(c[2], c[0], tdmap.shape[0:2], pathfinder=pf)
        maps.draw_agent  # noqa (marker drawn manually below)
        r = 6
        rgb[max(0, gx-r):gx+r, max(0, gy-r):gy+r] = [255, 0, 0]
        by_cat[cat] = by_cat.get(cat, 0) + 1
        n_marked += 1

    out = os.path.join(OUT, f"{SCENE_ID}_topdown.png")
    imageio.imwrite(out, rgb)
    print(f"marked {n_marked} goal objects on this floor: {by_cat}")
    print(f"saved {out}")
    sim.close()


if __name__ == "__main__":
    main()
