#!/usr/bin/env python3
"""Recreate the six-panel stress–strain plot from the archived CSV files."""

from pathlib import Path
import argparse
import csv
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ELEMENTS = ("Pt", "Au", "Ag")
AXES = ("111", "001")
COLORS = {"111": "#0000FF", "001": "#FF1F1F"}
Y_LIMITS = {"Pt": (0, 18.5), "Au": (0, 10), "Ag": (0, 10)}
STYLES = {1: "-", 2: "--", 3: ":", 4: "-.", 5: (0, (6, 2)), 6: (0, (3, 1, 1, 1))}


def load(path):
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    return (
        np.asarray([float(r["true_strain"]) for r in rows]),
        np.asarray([float(r["true_stress_GPa"]) for r in rows]),
    )


def smooth(values, width=9):
    pad = width // 2
    return np.convolve(np.pad(values, (pad, pad), mode="edge"), np.ones(width) / width, mode="valid")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/processed_stress_strain"))
    parser.add_argument("--output", type=Path, default=Path("stress_strain_all_realizations.png"))
    args = parser.parse_args()

    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "axes.labelweight": "bold"})
    fig, panels = plt.subplots(2, 3, figsize=(17, 9.2), sharex=True)
    for row, axis in enumerate(AXES):
        for col, element in enumerate(ELEMENTS):
            ax = panels[row, col]
            files = sorted(args.data.glob(f"{element}_6nm_{axis}_r*.csv"))
            for file in files:
                rep = int(re.search(r"_r(\d+)\.csv$", file.name).group(1))
                strain, stress = load(file)
                ax.plot(strain, smooth(stress), color=COLORS[axis], linestyle=STYLES[rep], lw=2, label=f"r{rep}")
            ax.set_title(f"{element} [{axis}]", fontweight="bold")
            ax.set_xlim(0, 0.25)
            ax.set_ylim(*Y_LIMITS[element])
            ax.grid(True, color="0.88", lw=0.6)
            ax.legend(frameon=False, ncol=2 if len(files) > 3 else 1)
            if row == 1:
                ax.set_xlabel(r"True Strain, $\epsilon_{\mathrm{T}}$")
            if col == 0:
                ax.set_ylabel(r"True Stress, $\sigma_{\mathrm{T}}$ (GPa)")
    fig.tight_layout()
    fig.savefig(args.output, dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
