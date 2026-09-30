import numpy as np
import pandas as pd
import pickle
import os
import glob
from typing import Optional

"""
StressCalculationV2.py  (Python 3.9 compatible)

What this script does:
- Loads contact areas from a per-realization pickle (contact_areas_v*.pkl).
- Parses per-timestep wall positions and forces from the per-realization LAMMPS log file.
- Merges on timestep, computes:
    * True strain:  ε_true = ln(L0 / L), where L = z_top - z_bottom
    * True stress: average of (|F_top|/A_top) and (|F_bottom|/A_bottom), converted to GPa
- Writes a per-realization pickle: stressvsstraindata_v2.pkl

How to run on the cluster:
    python StressCalculationV2.py
"""

# ----------------- USER CONFIG -----------------
BASE_DIR  = "/home/erabagomoreno/data/nano_data/6nm_7_strainrate_20_strain_sims_correctwallseperation"
ELEMENTS  = ("Ag", "Au", "Pt")
ORIENTS   = ("100", "111")
REPS      = (1, 2, 3)

# area pickle preference order (script will use the first one it finds)
AREA_PKL_CANDIDATES = ("contact_areas_v4.pkl", "contact_areas_v3.pkl", "contact_areas.pkl")

# output pickle written into each realization directory
OUT_NAME = "stressvsstraindata_v2.pkl"

# Conversion factor: (eV/Å^3) -> GPa
EV_PER_A3_TO_GPA = 160.21766208
# ----------------------------------------------


def read_log_forces_walls(log_path: str) -> pd.DataFrame:
    """
    Returns DataFrame with columns:
    timestep, force_top, force_bottom, z_top, z_bottom

    Assumptions (kept consistent with your V1 intent):
    - Lines of interest begin with a numeric timestep.
    - z_top and z_bottom are the last two numeric entries on the line.
    - force_top and force_bottom are the -4 and -3 numeric entries on the line.
    """
    rows = []
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            tok = line.strip().split()
            if not tok or not tok[0].isdigit():
                continue
            try:
                vals = [float(t) for t in tok]
            except ValueError:
                continue
            if len(vals) < 5:
                continue

            t = int(vals[0])
            z_top, z_bot = vals[-2], vals[-1]
            f_top, f_bot = vals[-4], vals[-3]
            rows.append((t, f_top, f_bot, z_top, z_bot))

    return pd.DataFrame(rows, columns=["timestep", "force_top", "force_bottom", "z_top", "z_bottom"])


def load_contact_areas_from_pkl(pkl_path: str) -> pd.DataFrame:
    """
    Loads the contact area pickle produced by your contact-area script (V3/V4).
    Expected keys: timestep, bottom_area, top_area
    Returns DataFrame with columns: timestep, top area, bottom area
    """
    with open(pkl_path, "rb") as f:
        payload = pickle.load(f)

    ts   = np.asarray(payload["timestep"])
    botA = np.asarray(payload["bottom_area"])
    topA = np.asarray(payload["top_area"])

    df = pd.DataFrame({
        "timestep":     ts.astype(np.int64),
        "top area":     topA.astype(np.float64),
        "bottom area":  botA.astype(np.float64),
    })
    return df


def build_stress_strain_pickle(areas_pkl: str, log_path: str, out_pkl: str) -> None:
    # Load inputs
    areas_df = load_contact_areas_from_pkl(areas_pkl)
    log_df   = read_log_forces_walls(log_path)

    # Join on timestep
    df = pd.merge(areas_df, log_df, on="timestep", how="inner").sort_values("timestep")
    if df.empty:
        raise SystemExit(
            f"No overlapping timesteps between contact areas ({areas_pkl}) and log file ({log_path})."
        )

    # True strain: ε_T = ln(L0 / L), with L = z_top - z_bottom
    L  = df["z_top"] - df["z_bottom"]
    L0 = L.iloc[0]
    df["true_strain"] = np.log(L0 / L)

    # Per-contact stress (eV/Å^3) -> GPa; use absolute forces
    with np.errstate(divide="ignore", invalid="ignore"):
        top_raw = np.where(
            df["top area"] > 0,
            np.abs(df["force_top"]) / df["top area"],
            np.nan
        )
        bottom_raw = np.where(
            df["bottom area"] > 0,
            np.abs(df["force_bottom"]) / df["bottom area"],
            np.nan
        )

    top_GPa    = top_raw * EV_PER_A3_TO_GPA
    bottom_GPa = bottom_raw * EV_PER_A3_TO_GPA

    # TOTAL true stress as the AVERAGE of top and bottom stresses
    df["true_stress_GPa"] = (top_GPa + bottom_GPa) / 2.0

    # Drop rows where total stress could not be computed
    df = df.dropna(subset=["true_stress_GPa"]).reset_index(drop=True)

    payload = {
        "timestep":        df["timestep"].to_numpy(dtype=np.int64),
        "true_strain":     df["true_strain"].to_numpy(dtype=np.float64),
        "true_stress_GPa": df["true_stress_GPa"].to_numpy(dtype=np.float64),
        "top area":        df["top area"].to_numpy(dtype=np.float64),
        "bottom area":     df["bottom area"].to_numpy(dtype=np.float64),
        "force_top":       df["force_top"].to_numpy(dtype=np.float64),
        "force_bottom":    df["force_bottom"].to_numpy(dtype=np.float64),
        "z_top":           df["z_top"].to_numpy(dtype=np.float64),
        "z_bottom":        df["z_bottom"].to_numpy(dtype=np.float64),
        "areas_pkl_path":  str(areas_pkl),
        "log_path":        str(log_path),
    }

    with open(out_pkl, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


def find_area_pickle(real_dir: str) -> Optional[str]:
    for name in AREA_PKL_CANDIDATES:
        p = os.path.join(real_dir, name)
        if os.path.isfile(p):
            return p
    # fallback: anything that looks like contact_areas*.pkl
    cands = sorted(glob.glob(os.path.join(real_dir, "contact_areas*.pkl")))
    return cands[0] if cands else None


def find_log_file(real_dir: str, element: str) -> Optional[str]:
    # Prefer expected naming
    expected = os.path.join(real_dir, f"log.{element}_shape")
    if os.path.isfile(expected):
        return expected
    # fallback: any log.* file
    cands = sorted(glob.glob(os.path.join(real_dir, "log.*")))
    return cands[0] if cands else None


def process_realization(real_dir: str, element: str) -> bool:
    areas_pkl = find_area_pickle(real_dir)
    if areas_pkl is None:
        print(f"[SKIP] No contact area pickle found in: {real_dir}")
        return False

    log_path = find_log_file(real_dir, element)
    if log_path is None:
        print(f"[SKIP] No log file found in: {real_dir}")
        return False

    out_pkl = os.path.join(real_dir, OUT_NAME)

    try:
        build_stress_strain_pickle(areas_pkl, log_path, out_pkl)
    except SystemExit as e:
        print(f"[SKIP] {real_dir}: {e}")
        return False
    except Exception as e:
        print(f"[ERROR] {real_dir}: {type(e).__name__}: {e}")
        return False

    print(f"[OK] Wrote {out_pkl}")
    return True


def main():
    n_ok = 0
    n_total = 0

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
