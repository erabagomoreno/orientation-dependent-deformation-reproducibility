#!/usr/bin/env python3
"""Build the compact, publication-facing reproducibility package."""

from __future__ import annotations

import csv
import pickle
import shutil
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
PAPER_DIR = ROOT / "Papers" / "OrientationDependentDeformation"
REPO = PAPER_DIR / "orientation-dependent-deformation-reproducibility"
FIGURE_ROOT = ROOT / "figure_recreation"
SOURCE_CODE = ROOT / "code"
VIDEO_ROOT = ROOT / "OrientationDependentDeformation" / "videos_of_sims" / "7strainratesims"

CASES = (
    ("Pt", "111", (1, 2, 3)),
    ("Pt", "100", (1, 2, 3)),
    ("Au", "111", (1, 2, 3)),
    ("Au", "100", (1, 2, 3)),
    ("Ag", "111", (1, 2, 3, 4, 5, 6)),
    ("Ag", "100", (1, 2, 3)),
)


def copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def export_curves() -> None:
    out = REPO / "data" / "processed_stress_strain"
    out.mkdir(parents=True, exist_ok=True)
    index_rows = []
    for element, orientation, reps in CASES:
        display = "001" if orientation == "100" else "111"
        for rep in reps:
            case = f"{element}_6nm_{orientation}_r{rep}"
            source = FIGURE_ROOT / "data" / case / "stressvsstraindata_v2.pkl"
            with source.open("rb") as stream:
                payload = pickle.load(stream)
            strain = np.asarray(payload["true_strain"], dtype=float)
            stress = np.asarray(payload["true_stress_GPa"], dtype=float)
            timestep = np.asarray(payload["timestep"], dtype=np.int64)
            destination = out / f"{element}_6nm_{display}_r{rep}.csv"
            with destination.open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["timestep", "true_strain", "true_stress_GPa"])
                writer.writerows(zip(timestep, strain, stress))
            index_rows.append((element, f"[{display}]", rep, len(strain), destination.name))

    with (REPO / "data" / "processed_stress_strain_index.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["material", "loading_orientation", "realization", "points", "file"])
        writer.writerows(index_rows)


def write_text_files() -> None:
    (REPO / "data").mkdir(parents=True, exist_ok=True)
    (REPO / "videos").mkdir(parents=True, exist_ok=True)
    readme = """# Orientation-dependent deformation of noble-metal nanoparticles

This repository is the compact reproducibility archive accompanying the manuscript and supplementary information on orientation-dependent deformation of Pt, Au, and Ag nanoparticles. It contains processed stress–strain data, analysis scripts, summary calculations, publication plots, and representative atomistic visualization videos.

## Contents

- `data/processed_stress_strain/`: CSV exports for 21 molecular-dynamics realizations. Pt and Au have three realizations per orientation; Ag [001] has three; Ag [111] has six.
- `analysis/yield_crss_summary.xlsx`: auditable table of experimental and simulated yield strengths, Schmid factors, experimental CRSS, SFE, and deformation modes.
- `analysis/yield_strengths.csv`: realization-level yield values and group statistics used for the simulation summary.
- `scripts/`: contact-area, true-stress/true-strain, data-export, and plotting code.
- `plots/`: publication-ready PNG and PDF outputs.
- `videos/`: one representative yield visualization for each material/orientation case.
- `lammps_examples/`: compact log/input provenance examples for the 6 nm and 20 nm Pt calculations.

## Data provenance

The full trajectories are retained on the research cluster because individual dump series are too large for a source repository. The processed 6 nm data derive from:

`/home/erabagomoreno/data/nano_data/6nm_7_strainrate_20_strain_sims_correctwallseperation`

The qualitative 20 nm Pt [001] size-check derives from:

`/home/erabagomoreno/data/nano_data/20nm_7_strainrate_20_strain_sims_correctwallseperation/Pt_20nm_100_r1`

That run contains approximately 287,139 atoms and was used to verify that twinning persists at a particle size closer to the experiments. It is not included in the six-case yield-strength averages.

## Reproducing the plots

Create a Python environment with NumPy and Matplotlib, then run:

```bash
python scripts/recreate_figures.py --base data/pickle_examples --output-dir reproduced_plots
```

The repository distributes CSVs as the archival, software-neutral data format. The plotting script documents the original pickle-based workflow; `scripts/export_processed_curves.py` shows how the CSVs were produced.

## Analysis definitions

- True strain: `ln(L0/L)`, where `L` is the instantaneous wall separation.
- True stress: the mean of the absolute top- and bottom-wall forces divided by their independently calculated contact areas, converted from eV/Å³ to GPa.
- Simulation yield strength: the true stress immediately before the first stress drop; in the archived calculation this corresponds to the maximum of each processed curve.
- Error-bar extent: minimum to maximum yield strength among independent realizations.
- Experimental CRSS: experimental yield strength multiplied by the Schmid factor of the operative mechanism (`m = 0.2722` for [111] slip; `m = 0.4714` for [001] twinning).

## Notes

The workbook identifies current manuscript-draft values separately from simulation-derived values. No uncertainty has been invented where the draft does not report one.
"""
    (REPO / "README.md").write_text(readme)
    (REPO / ".gitignore").write_text(".DS_Store\n__pycache__/\n*.pyc\n~$*.xlsx\n")
    (REPO / "data" / "README.md").write_text(
        "Processed CSVs are derived from the per-realization `stressvsstraindata_v2.pkl` files. "
        "The raw trajectory series remain on the cluster and are not duplicated here.\n"
    )
    (REPO / "videos" / "README.md").write_text(
        "Representative yield-event visualizations. File names use `001` for simulation folders named `100`.\n"
    )


def assemble() -> None:
    REPO.mkdir(parents=True, exist_ok=True)
    export_curves()
    write_text_files()

    scripts = {
        SOURCE_CODE / "contact_area_calculationV4.py": "contact_area_calculation.py",
        SOURCE_CODE / "StressCalculationV2.py": "calculate_true_stress_strain.py",
        FIGURE_ROOT / "recreate_figures.py": "recreate_figures.py",
        FIGURE_ROOT / "process_ag111_realization.py": "process_realization.py",
    }
    for source, name in scripts.items():
        copy(source, REPO / "scripts" / name)
    copy(Path(__file__), REPO / "scripts" / "export_processed_curves.py")

    for name in (
        "simulation_all_realizations_2x3.png",
        "simulation_all_realizations_2x3.pdf",
        "simulation_stress_strain_yield_summary.png",
        "simulation_stress_strain_yield_summary.pdf",
    ):
        copy(FIGURE_ROOT / "output" / name, REPO / "plots" / name)
    copy(FIGURE_ROOT / "output" / "yield_strengths.csv", REPO / "analysis" / "yield_strengths.csv")

    videos = {
        "pt_111_yield.mp4": "Pt_111_yield.mp4",
        "pt_100_yield.mp4": "Pt_001_yield.mp4",
        "au_111_yield.mp4": "Au_111_yield.mp4",
        "au_100_yield.mp4": "Au_001_yield.mp4",
        "ag_111_yield.mp4": "Ag_111_yield.mp4",
        "ag_100_yield.mp4": "Ag_001_yield.mp4",
    }
    for source_name, destination_name in videos.items():
        copy(VIDEO_ROOT / source_name, REPO / "videos" / destination_name)


if __name__ == "__main__":
    assemble()
    print(REPO)
