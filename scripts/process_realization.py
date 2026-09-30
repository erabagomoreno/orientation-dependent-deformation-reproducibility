#!/usr/bin/env python3
"""Generate contact-area and stress-strain pickles for one Ag [111] run.

This applies the same definitions used by contact_area_calculationV4.py and
StressCalculationV2.py to a single realization. Raw trajectory and log files
are read only; the two derived pickle files are written beside them.
"""

from __future__ import annotations

import argparse
import pickle
import re
from pathlib import Path

import numpy as np


EV_PER_A3_TO_GPA = 160.21766208
CUTOFF_A = 3.0


def read_log(log_path: Path) -> dict[int, tuple[float, float, float, float]]:
    """Return timestep -> (force_top, force_bottom, z_top, z_bottom)."""
    rows: dict[int, tuple[float, float, float, float]] = {}
    with log_path.open("r", encoding="utf-8", errors="ignore") as stream:
        for line in stream:
            tokens = line.split()
            if not tokens or not tokens[0].isdigit():
                continue
            try:
                values = [float(token) for token in tokens]
            except ValueError:
                continue
            if len(values) < 5:
                continue
            rows[int(values[0])] = (values[-4], values[-3], values[-2], values[-1])
    if not rows:
        raise RuntimeError(f"No force/wall records parsed from {log_path}")
    return rows


def parse_frame(path: Path) -> tuple[int | None, np.ndarray]:
    timestep = None
    columns: list[str] = []
    data: list[list[str]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as stream:
        while True:
            line = stream.readline()
            if not line:
                break
            if line.startswith("ITEM: TIMESTEP"):
                timestep = int(stream.readline())
            elif line.startswith("ITEM: ATOMS"):
                columns = line.split()[2:]
                for row in stream:
                    if row.startswith("ITEM:"):
                        break
                    data.append(row.split())
                break

    def column_index(name: str, fallback_one_based: int) -> int:
        return columns.index(name) if name in columns else fallback_one_based - 1

    ix = column_index("x", 3)
    iy = column_index("y", 4)
    iz = column_index("z", 5)
    positions = np.asarray(
        [[float(row[ix]), float(row[iy]), float(row[iz])] for row in data if len(row) > iz],
        dtype=float,
    )
    return timestep, positions


def convex_hull_area(points: np.ndarray) -> float:
    if len(points) < 3:
        return 0.0
    unique = np.unique(points, axis=0)
    if len(unique) < 3:
        return 0.0
    ordered = unique[np.lexsort((unique[:, 1], unique[:, 0]))]

    def cross(origin: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
        return float((a[0] - origin[0]) * (b[1] - origin[1]) - (a[1] - origin[1]) * (b[0] - origin[0]))

    lower: list[np.ndarray] = []
    for point in ordered:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper: list[np.ndarray] = []
    for point in ordered[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    hull = np.vstack((lower[:-1], upper[:-1]))
    x, y = hull[:, 0], hull[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def frame_number(path: Path) -> int:
    match = re.search(r"\.(\d+)\.lammpstrj$", path.name)
    return int(match.group(1)) if match else -1


def process(folder: Path) -> None:
    log_path = folder / "log.Ag_shape"
    log_rows = read_log(log_path)
    frames = sorted(folder.glob("Ag_6nm_shape.*.lammpstrj"), key=frame_number)
    if not frames:
        raise RuntimeError(f"No trajectory frames found in {folder}")

    timesteps: list[int] = []
    bottom_areas: list[float] = []
    top_areas: list[float] = []
    forces_top: list[float] = []
    forces_bottom: list[float] = []
    z_top_values: list[float] = []
    z_bottom_values: list[float] = []

    for index, frame in enumerate(frames, start=1):
        timestep, positions = parse_frame(frame)
        if timestep is None or timestep not in log_rows or positions.size == 0:
            continue
        force_top, force_bottom, z_top, z_bottom = log_rows[timestep]
        z = positions[:, 2]
        top_xy = positions[(z <= z_top) & ((z_top - z) <= CUTOFF_A), :2]
        bottom_xy = positions[(z >= z_bottom) & ((z - z_bottom) <= CUTOFF_A), :2]
        timesteps.append(timestep)
        top_areas.append(convex_hull_area(top_xy))
        bottom_areas.append(convex_hull_area(bottom_xy))
        forces_top.append(force_top)
        forces_bottom.append(force_bottom)
        z_top_values.append(z_top)
        z_bottom_values.append(z_bottom)
        if index % 500 == 0:
            print(f"[{folder.name}] processed {index}/{len(frames)} frames", flush=True)

    ts = np.asarray(timesteps, dtype=np.int64)
    order = np.argsort(ts, kind="stable")
    ts = ts[order]
    top_area = np.asarray(top_areas, dtype=float)[order]
    bottom_area = np.asarray(bottom_areas, dtype=float)[order]
    force_top = np.asarray(forces_top, dtype=float)[order]
    force_bottom = np.asarray(forces_bottom, dtype=float)[order]
    z_top = np.asarray(z_top_values, dtype=float)[order]
    z_bottom = np.asarray(z_bottom_values, dtype=float)[order]

    area_payload = {
        "timestep": ts,
        "bottom_area": bottom_area,
        "top_area": top_area,
        "log_path": str(log_path),
        "real_dir": str(folder),
        "cutoff_A": CUTOFF_A,
    }
    area_path = folder / "contact_areas_v4.pkl"
    with area_path.open("wb") as stream:
        pickle.dump(area_payload, stream, protocol=pickle.HIGHEST_PROTOCOL)

    separation = z_top - z_bottom
    true_strain = np.log(separation[0] / separation)
    with np.errstate(divide="ignore", invalid="ignore"):
        stress_top = np.where(top_area > 0, np.abs(force_top) / top_area, np.nan)
        stress_bottom = np.where(bottom_area > 0, np.abs(force_bottom) / bottom_area, np.nan)
    true_stress = 0.5 * (stress_top + stress_bottom) * EV_PER_A3_TO_GPA
    valid = np.isfinite(true_stress)
    stress_payload = {
        "timestep": ts[valid],
        "true_strain": true_strain[valid],
        "true_stress_GPa": true_stress[valid],
        "top area": top_area[valid],
        "bottom area": bottom_area[valid],
        "force_top": force_top[valid],
        "force_bottom": force_bottom[valid],
        "z_top": z_top[valid],
        "z_bottom": z_bottom[valid],
        "areas_pkl_path": str(area_path),
        "log_path": str(log_path),
    }
    stress_path = folder / "stressvsstraindata_v2.pkl"
    with stress_path.open("wb") as stream:
        pickle.dump(stress_payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[OK] {folder.name}: {valid.sum()} stress-strain points", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--rep", type=int, choices=(4, 5, 6), required=True)
    args = parser.parse_args()
    process(args.base / f"Ag_6nm_111_r{args.rep}")


if __name__ == "__main__":
    main()
