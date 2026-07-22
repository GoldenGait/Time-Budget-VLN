"""Phase 1 - build P(room | object) from MP3D house_segmentations (.house files).

Counted, category-level (mpcat40), scene-independent -> transfers to HM3D via
observed-object voting (Phase 2). Deterministic: sorted iteration, fixed epsilon.

  python build_prior.py --house-root /data/maitree-tiamat/mp3d_semantics \
                        --out datagen/prior_table.json
"""
import argparse, glob, json, os, collections

# Official MP3D region label codes -> room names.
ROOM = {
    'a': 'bathroom', 'b': 'bedroom', 'c': 'closet', 'd': 'dining_room',
    'e': 'entryway', 'f': 'family_room', 'g': 'garage', 'h': 'hallway',
    'i': 'library', 'j': 'laundry_room', 'k': 'kitchen', 'l': 'living_room',
    'm': 'meeting_room', 'n': 'lounge', 'o': 'office', 'p': 'porch',
    'r': 'rec_room', 's': 'stairs', 't': 'toilet_room', 'u': 'utility_room',
    'v': 'tv_room', 'w': 'gym', 'x': 'outdoor', 'y': 'balcony', 'z': 'other_room',
    'B': 'bar', 'C': 'classroom', 'D': 'dining_booth', 'S': 'spa', 'Z': 'junk',
}
ROOMS = sorted(set(ROOM.values()))          # fixed room vocabulary
EPS = 0.02                                   # floor so no (cat,room) is exactly zero


def parse_house(path):
    """Return list of (mpcat40_name, room_name) for every object instance."""
    region_room, cat_mp = {}, {}
    for ln in open(path, errors='ignore'):
        t = ln.split()
        if not t:
            continue
        if t[0] == 'R':                      # R idx level 0 0 LABEL ...
            region_room[t[1]] = ROOM.get(t[5], 'other_room')
        elif t[0] == 'C':                    # C cat_idx map_idx raw mpcat40_idx mpcat40_name ...
            # anchor on mpcat40_idx being an int at t[4]; if raw name had spaces this shifts,
            # so scan for the first int >=t[4] position and take the following token.
            if len(t) >= 6 and t[4].lstrip('-').isdigit():
                cat_mp[t[1]] = t[5]
    out = []
    for ln in open(path, errors='ignore'):
        t = ln.split()
        if t and t[0] == 'O':                # O obj_idx region_idx cat_idx ...
            room = region_room.get(t[2])
            mp = cat_mp.get(t[3])
            if room and mp:
                out.append((mp, room))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--house-root', default='/data/maitree-tiamat/mp3d_semantics')
    ap.add_argument('--out', default='datagen/prior_table.json')
    a = ap.parse_args()

    houses = sorted(glob.glob(os.path.join(a.house_root, '*/*/house_segmentations/*.house')))
    counts = collections.defaultdict(lambda: collections.Counter())
    n_obj = 0
    for h in houses:
        for mp, room in parse_house(h):
            counts[mp][room] += 1
            n_obj += 1

    # P(room|cat) with epsilon floor over the FIXED room vocabulary, then renormalize.
    prior = {}
    for cat in sorted(counts):
        row = {r: counts[cat].get(r, 0) + EPS for r in ROOMS}
        s = sum(row.values())
        prior[cat] = {r: row[r] / s for r in ROOMS}

    meta = {'source': 'MP3D house_segmentations (counted)', 'n_scenes': len(houses),
            'n_object_instances': n_obj, 'epsilon': EPS, 'rooms': ROOMS,
            'n_categories': len(prior)}
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    json.dump({'meta': meta, 'prior': prior}, open(a.out, 'w'), indent=2, sort_keys=True)

    # -------- GATE P1 --------
    print(f"[P1] scenes={len(houses)} objects={n_obj} categories={len(prior)} rooms={len(ROOMS)}")
    bad = [c for c in prior if abs(sum(prior[c].values()) - 1.0) > 1e-9]
    print(f"[P1] rows summing to 1: {len(prior)-len(bad)}/{len(prior)}  (bad={bad[:5]})")
    print("[P1] spot-check (top-3 rooms per goal category):")
    for g in ['toilet', 'bed', 'tv_monitor', 'plant', 'chair', 'sofa']:
        if g in prior:
            top = sorted(prior[g].items(), key=lambda kv: -kv[1])[:3]
            print(f"   {g:12s} -> " + ", ".join(f"{r}={p:.2f}" for r, p in top))
        else:
            print(f"   {g:12s} -> NOT IN TABLE (mpcat40 name mismatch?)")
    print(f"[P1] wrote {a.out}")


if __name__ == '__main__':
    main()
