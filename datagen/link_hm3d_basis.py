"""One-time setup: the HM3D-Sem download names assets <stem>.glb / <stem>.semantic.glb /
<stem>.semantic.txt, but the annotated_basis scene-dataset config globs for `*.basis.glb`
and derives `<stem>.basis.semantic.{glb,txt}`. Create the `.basis.*` symlinks so the config
resolves and the semantic sensor / semantic_scene load. Idempotent.

  python link_hm3d_basis.py --hm3d-root <.../hm3d-0.2/hm3d> --splits train
"""
import argparse, glob, os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hm3d-root",
                    default="/data/maitree-tiamat/navila/scene_datasets/versioned_data/hm3d-0.2/hm3d")
    ap.add_argument("--splits", nargs="+", default=["train"])
    a = ap.parse_args()
    made = 0
    for split in a.splits:
        for semglb in sorted(glob.glob(os.path.join(a.hm3d_root, split, "*", "*.semantic.glb"))):
            if ".basis." in os.path.basename(semglb):
                continue
            d = os.path.dirname(semglb)
            stem = os.path.basename(semglb)[: -len(".semantic.glb")]
            links = [(f"{stem}.glb", f"{stem}.basis.glb"),
                     (f"{stem}.semantic.glb", f"{stem}.basis.semantic.glb"),
                     (f"{stem}.semantic.txt", f"{stem}.basis.semantic.txt")]
            for src, dst in links:
                dp = os.path.join(d, dst)
                if not os.path.exists(dp) and os.path.exists(os.path.join(d, src)):
                    os.symlink(src, dp)
                    made += 1
    print(f"created {made} symlinks across splits={a.splits}")


if __name__ == "__main__":
    main()
