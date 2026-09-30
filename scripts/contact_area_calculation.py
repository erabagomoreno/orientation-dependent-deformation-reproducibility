import numpy as np
import glob
import os
import re
import pickle

"""
contact_area_calculationV4.py

What this script does (same core behavior as V3):
- Reads wall positions (z_top, z_bot) vs timestep from a LAMMPS log file.
- For each single-frame trajectory dump (*.lammpstrj), it:
    * reads atom positions for that timestep
    * selects atoms within CUTOFF_A of the top wall and bottom wall
    * computes the 2D convex-hull area in XY of those selected atoms
- Saves a pickle file with arrays: timestep, bottom_area, top_area

What changed in V4:
- Automatically discovers and processes ALL 18 realizations under:
  /home/erabagomoreno/data/nano_data/6nm_7_strainrate_20_strain_sims_correctwallseperation/
  for elements {Ag, Au, Pt}, orientations {100, 111}, and realizations r{1,2,3}.
- Writes one pickle output *inside each realization folder*.
"""

# --------- USER CONFIG ---------
BASE_DIR  = "/home/erabagomoreno/data/nano_data/6nm_7_strainrate_20_strain_sims_correctwallseperation"
ELEMENTS  = ("Ag", "Au", "Pt")
ORIENTS   = ("100", "111")
REPS      = (1, 2, 3)

# atoms within this distance (Å) from each wall are counted as "in contact"
CUTOFF_A  = 3.0

# output pickle name written into each realization directory
OUT_NAME  = "contact_areas_v4.pkl"
# -------------------------------


def read_log_wall_positions(log_path):
    """Returns dict: timestep -> (z_top, z_bot)."""
    wall_by_t = {}
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            tok = line.strip().split()
            if len(tok) < 3:
                continue
            if tok[0].isdigit():
                try:
                    vals = [float(t) for t in tok]
                    t = int(vals[0])
                    z_top, z_bot = vals[-2], vals[-1]
                    wall_by_t[t] = (z_top, z_bot)
                except Exception:
                    continue
    return wall_by_t


def parse_single_frame(path):
    """Returns (timestep, xyz array Nx3) for one dump file that contains one frame."""
    ts = None
    data = []
    cols = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        while True:
            line = f.readline()
            if not line:
                break
            if line.startswith("ITEM: TIMESTEP"):
                ts = int(f.readline().strip())
            elif line.startswith("ITEM: ATOMS"):
                cols = line.strip().split()[2:]
                for row in f:
                    if row.startswith("ITEM:"):
                        break
                    parts = row.split()
                    data.append(parts)
                break

    def idx(name, fallback):
        return cols.index(name) if name in cols else (fallback - 1)

    # LAMMPS dump columns are usually: id type x y z ...
    ix, iy, iz = idx('x', 3), idx('y', 4), idx('z', 5)
    arr = np.array([[float(r[ix]), float(r[iy]), float(r[iz])]
                    for r in data if len(r) > iz], dtype=float)
    return ts, arr


def convex_hull_area_xy(xy):
    """Monotone chain convex hull area in XY."""
    if len(xy) < 3:
        return 0.0

    pts = xy[np.lexsort((xy[:, 1], xy[:, 0]))]

    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper = []
    for p in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    hull = np.vstack((lower[:-1], upper[:-1]))
    if len(hull) < 3:
        return 0.0

    x, y = hull[:, 0], hull[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def areas_for_file(path, walls_dict, cutoff):
    """Compute (timestep, bottom_area, top_area) for one dump file."""
    ts, position = parse_single_frame(path)
    if ts is None or ts not in walls_dict or position.size == 0:
        return None

    z_top, z_bot = walls_dict[ts]
    z = position[:, 2]

    # within cutoff Å from the walls
    top_sel    = (z <= z_top) & ((z_top - z) <= cutoff)
    bottom_sel = (z >= z_bot) & ((z - z_bot) <= cutoff)

    top_xy    = position[top_sel][:, :2]
    bottom_xy = position[bottom_sel][:, :2]

    top_area    = convex_hull_area_xy(top_xy) if len(top_xy) else 0.0
    bottom_area = convex_hull_area_xy(bottom_xy) if len(bottom_xy) else 0.0
    return (ts, bottom_area, top_area)


def extract_step_from_name(p):
    """Use the last integer in the basename (assumed timestep)."""
    nums = re.findall(r'\d+', os.path.basename(p))
    return int(nums[-1]) if nums else -1


def find_log_file(real_dir, element):
    """
    Prefer expected naming: log.<element>_shape
    Fallback: any file starting with 'log.' inside real_dir.
    """
    expected = os.path.join(real_dir, f"log.{element}_shape")
    if os.path.isfile(expected):
        return expected

    candidates = sorted(glob.glob(os.path.join(real_dir, "log.*")))
    if candidates:
        return candidates[0]

    return None


def find_trj_files(real_dir, element):
    """
    Prefer expected naming: <element>_6nm_shape.*.lammpstrj
    Fallback: any *.lammpstrj inside real_dir.
    """
    preferred = sorted(glob.glob(os.path.join(real_dir, f"{element}_6nm_shape.*.lammpstrj")),
                       key=extract_step_from_name)
    if preferred:
        return preferred

    fallback = sorted(glob.glob(os.path.join(real_dir, "*.lammpstrj")),
                      key=extract_step_from_name)
    return fallback


def process_realization(real_dir, element):
    """Process one realization folder and write OUT_NAME into it."""
    log_path = find_log_file(real_dir, element)
    if log_path is None:
        print(f"[SKIP] No log file found in: {real_dir}")
        return False

    trj_files = find_trj_files(real_dir, element)
    if not trj_files:
        print(f"[SKIP] No .lammpstrj files found in: {real_dir}")
        return False

    walls = read_log_wall_positions(log_path)
    if not walls:
        print(f"[SKIP] Wall positions not parsed (empty) from: {log_path}")
        return False

    rows = []
    for fp in trj_files:
        result = areas_for_file(fp, walls, CUTOFF_A)
        if result is not None:
            rows.append(result)

    if not rows:
        print(f"[SKIP] No valid frames produced areas in: {real_dir}")
        return False

    payload = {
        "timestep":     np.array([r[0] for r in rows], dtype=np.int64),
        "bottom_area":  np.array([r[1] for r in rows], dtype=np.float64),
        "top_area":     np.array([r[2] for r in rows], dtype=np.float64),
        "log_path":     log_path,
        "real_dir":     real_dir,
        "cutoff_A":     float(CUTOFF_A),
    }

    out_pkl = os.path.join(real_dir, OUT_NAME)
    with open(out_pkl, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"[OK] Wrote {out_pkl} ({len(rows)} frames)")
    return True


def main():
    n_ok = 0
    n_total = 0

    # Expected folder naming: <element>_6nm_<orient>_r<rep>
    for el in ELEMENTS:
        for ori in ORIENTS:
            for r in REPS:
                n_total += 1
                real_dir = os.path.join(BASE_DIR, f"{el}_6nm_{ori}_r{r}")
                if not os.path.isdir(real_dir):
                    print(f"[MISSING] {real_dir}")
                    continue
                if process_realization(real_dir, el):
                    n_ok += 1

    print(f"\nDone. Successful: {n_ok}/{n_total}")


if __name__ == "__main__":
    main()
