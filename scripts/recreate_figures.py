#!/usr/bin/env python3
"""Recreate the simulation summary and all-realization figures."""

from __future__ import annotations

import argparse
import csv
import pickle
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator


ELEMENTS = ("Pt", "Au", "Ag")
ORIENTATIONS = ("111", "100")
DISPLAY_ORIENTATION = {"111": "111", "100": "001"}
COLORS = {"111": "#0000FF", "100": "#FF1F1F"}
ELEMENT_STYLES = {"Pt": "-", "Au": (0, (7, 7)), "Ag": (0, (1.2, 4.2))}
REALIZATION_STYLES = {
    1: "-",
    2: "--",
    3: ":",
    4: "-.",
    5: (0, (6, 2)),
    6: (0, (3, 1, 1, 1)),
}
EXPECTED_REPS = {
    (element, orientation): (1, 2, 3)
    for element in ELEMENTS
    for orientation in ORIENTATIONS
}
EXPECTED_REPS[("Ag", "111")] = (1, 2, 3, 4, 5, 6)

SMOOTH_WINDOW = 9
X_LIMITS = (0.0, 0.25)
SUMMARY_X_LIMITS = (0.0, 0.30)
Y_LIMITS_BY_ELEMENT = {"Pt": (0.0, 18.5), "Au": (0.0, 10.0), "Ag": (0.0, 10.0)}


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        # The reference uses Computer Modern math glyphs; this matters most
        # for the lunate epsilon in the true-strain variable.
        "mathtext.fontset": "cm",
        "font.weight": "bold",
        "axes.labelweight": "bold",
        "axes.titleweight": "bold",
        "axes.linewidth": 1.0,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": False,
        "xtick.major.width": 0.9,
        "ytick.major.width": 0.9,
        "xtick.minor.width": 0.7,
        "ytick.minor.width": 0.7,
        "legend.frameon": False,
        "savefig.facecolor": "white",
    }
)


def moving_average(values: np.ndarray, window: int = SMOOTH_WINDOW) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if window <= 1 or values.size < 3:
        return values.copy()
    width = max(3, int(window))
    if width % 2 == 0:
        width += 1
    pad = width // 2
    return np.convolve(
        np.pad(values, (pad, pad), mode="edge"), np.ones(width) / width, mode="valid"
    )


def load_series(base: Path, element: str, orientation: str, rep: int) -> dict[str, np.ndarray]:
    path = base / f"{element}_6nm_{orientation}_r{rep}" / "stressvsstraindata_v2.pkl"
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("rb") as stream:
        payload = pickle.load(stream)
    strain = np.asarray(payload["true_strain"], dtype=float).reshape(-1)
    stress = np.asarray(payload["true_stress_GPa"], dtype=float).reshape(-1)
    valid = np.isfinite(strain) & np.isfinite(stress)
    strain, stress = strain[valid], stress[valid]
    order = np.argsort(strain, kind="stable")
    strain, stress = strain[order], stress[order]
    if strain.size == 0:
        raise ValueError(f"No finite data in {path}")
    return {"strain": strain, "stress": stress, "smooth": moving_average(stress), "path": path}


def load_all(base: Path) -> dict[tuple[str, str, int], dict[str, np.ndarray]]:
    data: dict[tuple[str, str, int], dict[str, np.ndarray]] = {}
    missing: list[str] = []
    for element in ELEMENTS:
        for orientation in ORIENTATIONS:
            for rep in EXPECTED_REPS[(element, orientation)]:
                try:
                    data[(element, orientation, rep)] = load_series(base, element, orientation, rep)
                except FileNotFoundError as exc:
                    missing.append(str(exc))
    if missing:
        raise SystemExit("Missing required processed data:\n  " + "\n  ".join(missing))
    return data


def yield_point(series: dict[str, np.ndarray]) -> tuple[float, float]:
    index = int(np.nanargmax(series["stress"]))
    return float(series["strain"][index]), float(series["stress"][index])


def style_curve_axis(
    ax: plt.Axes,
    ylimits: tuple[float, float],
    labelsize: int = 13,
    xlimits: tuple[float, float] = X_LIMITS,
) -> None:
    ax.set_xlim(*xlimits)
    ax.set_ylim(*ylimits)
    ax.set_xticks([0.00, 0.05, 0.10, 0.15, 0.20, 0.25])
    ax.set_xticklabels(["0", "0.05", "0.10", "0.15", "0.2", "0.25"])
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(which="major", labelsize=labelsize, length=4.5)
    ax.tick_params(which="minor", length=2.5)
    ax.grid(True, which="major", axis="both", color="0.88", linewidth=0.60)
    ax.grid(False, which="minor")
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")


def summarize_yields(data: dict) -> dict[tuple[str, str], dict[str, object]]:
    summary: dict[tuple[str, str], dict[str, object]] = {}
    for element in ELEMENTS:
        for orientation in ORIENTATIONS:
            values = np.asarray(
                [yield_point(data[(element, orientation, rep)])[1] for rep in EXPECTED_REPS[(element, orientation)]],
                dtype=float,
            )
            summary[(element, orientation)] = {
                "values": values,
                "mean": float(np.mean(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            }
    return summary


def create_summary(data: dict, output_dir: Path) -> None:
    summary = summarize_yields(data)
    fig, (ax_curve, ax_bar) = plt.subplots(
        1, 2, figsize=(12.0, 4.9), gridspec_kw={"width_ratios": [1.08, 1.0]}
    )

    style_curve_axis(ax_curve, (0.0, 18.5), labelsize=16.1, xlimits=SUMMARY_X_LIMITS)
    ax_curve.set_xticks([0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30])
    ax_curve.set_xticklabels(["0", "0.05", "0.10", "0.15", "0.2", "0.25", "0.3"])
    ax_curve.set_xlabel(r"True Strain, $\epsilon_{\mathrm{T}}$", fontsize=21.85, fontweight="bold")
    ax_curve.set_ylabel(r"True Stress, $\sigma_T$ (GPa)", fontsize=21.85, fontweight="bold")
    handles: list[Line2D] = []
    for element in ELEMENTS:
        for orientation in ORIENTATIONS:
            series = data[(element, orientation, 1)]
            ax_curve.plot(
                series["strain"],
                series["smooth"],
                color=COLORS[orientation],
                linestyle=ELEMENT_STYLES[element],
                linewidth=2.0,
            )
            yield_strain, _ = yield_point(series)
            marker_index = int(np.argmin(np.abs(series["strain"] - yield_strain)))
            ax_curve.plot(
                series["strain"][marker_index],
                series["smooth"][marker_index],
                "o",
                markersize=7.5,
                markerfacecolor="black",
                markeredgecolor="black",
                zorder=8,
            )
            handles.append(
                Line2D(
                    [0],
                    [0],
                    color=COLORS[orientation],
                    linestyle=ELEMENT_STYLES[element],
                    linewidth=2.0,
                    label=rf"{element}$_{{{DISPLAY_ORIENTATION[orientation]}}}$",
                )
            )
    legend = ax_curve.legend(
        handles=handles,
        loc="center right",
        fontsize=16.1,
        handlelength=2.4,
        borderpad=0.2,
        labelspacing=0.22,
        handletextpad=0.6,
    )
    for label in legend.get_texts():
        label.set_fontweight("bold")

    x = np.arange(len(ELEMENTS), dtype=float)
    width = 0.20
    gap = 0.045
    center_offset = (width + gap) / 2.0
    x111, x001 = x - center_offset, x + center_offset
    means111 = np.asarray([summary[(element, "111")]["mean"] for element in ELEMENTS])
    means001 = np.asarray([summary[(element, "100")]["mean"] for element in ELEMENTS])
    mins111 = np.asarray([summary[(element, "111")]["min"] for element in ELEMENTS])
    mins001 = np.asarray([summary[(element, "100")]["min"] for element in ELEMENTS])
    maxs111 = np.asarray([summary[(element, "111")]["max"] for element in ELEMENTS])
    maxs001 = np.asarray([summary[(element, "100")]["max"] for element in ELEMENTS])
    err111 = np.vstack((means111 - mins111, maxs111 - means111))
    err001 = np.vstack((means001 - mins001, maxs001 - means001))

    bar_style = dict(edgecolor="black", linewidth=0.8, capsize=2.5, ecolor="black")
    ax_bar.bar(x111, means111, width=width, color=COLORS["111"], yerr=err111, label="[111]", **bar_style)
    ax_bar.bar(x001, means001, width=width, color=COLORS["100"], yerr=err001, label="[001]", **bar_style)
    ax_bar.set_ylabel("Yield Strength (GPa)", fontsize=21.85, fontweight="bold")
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(ELEMENTS)
    ax_bar.set_ylim(0.0, 20.0)
    ax_bar.yaxis.set_minor_locator(AutoMinorLocator())
    ax_bar.tick_params(which="major", labelsize=16.1, length=4.5)
    ax_bar.tick_params(which="minor", length=2.5)
    ax_bar.grid(True, which="major", axis="both", color="0.88", linewidth=0.60)
    ax_bar.set_axisbelow(True)
    for label in ax_bar.get_xticklabels() + ax_bar.get_yticklabels():
        label.set_fontweight("bold")
    bar_legend = ax_bar.legend(
        loc="upper right",
        fontsize=16.1,
        handlelength=1.8,
        handleheight=1.0,
        borderpad=0.2,
        labelspacing=0.20,
        handletextpad=0.45,
    )
    for label in bar_legend.get_texts():
        label.set_fontweight("bold")

    for index, element in enumerate(ELEMENTS):
        drop = 100.0 * (means111[index] - means001[index]) / means111[index]
        text_y = maxs001[index] + (0.42 if element == "Pt" else 0.30)
        ax_bar.text(
            x001[index] + 0.02,
            text_y,
            f"-{drop:.0f}%",
            ha="center",
            va="bottom",
            fontsize=16.1,
            fontweight="bold",
            zorder=7,
        )

    ax_curve.text(-0.13, 1.03, "(a)", transform=ax_curve.transAxes, fontsize=19.55, fontweight="bold", va="top")
    ax_bar.text(-0.22, 1.03, "(b)", transform=ax_bar.transAxes, fontsize=19.55, fontweight="bold", va="top")
    fig.subplots_adjust(left=0.082, right=0.985, bottom=0.17, top=0.96, wspace=0.25)
    fig.savefig(output_dir / "simulation_stress_strain_yield_summary.png", dpi=600)
    fig.savefig(output_dir / "simulation_stress_strain_yield_summary.pdf")
    plt.close(fig)


def create_realizations(data: dict, output_dir: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(17.0, 9.1), sharex=True, sharey=False)
    for row, orientation in enumerate(ORIENTATIONS):
        for column, element in enumerate(ELEMENTS):
            ax = axes[row, column]
            style_curve_axis(ax, Y_LIMITS_BY_ELEMENT[element], labelsize=12)
            ax.set_title(
                f"{element} [{DISPLAY_ORIENTATION[orientation]}]",
                fontsize=15,
                fontweight="bold",
                pad=7,
            )
            reps = EXPECTED_REPS[(element, orientation)]
            for rep in reps:
                series = data[(element, orientation, rep)]
                ax.plot(
                    series["strain"],
                    series["smooth"],
                    color=COLORS[orientation],
                    linewidth=1.7,
                    linestyle=REALIZATION_STYLES[rep],
                    label=f"r{rep}",
                )
            legend = ax.legend(
                loc="upper right",
                fontsize=10.5,
                handlelength=2.8,
                borderpad=0.15,
                labelspacing=0.20,
                handletextpad=0.45,
                ncol=2 if len(reps) > 3 else 1,
                columnspacing=0.8,
            )
            for label in legend.get_texts():
                label.set_fontweight("bold")
            if row == 1:
                ax.set_xlabel(r"True Strain, $\epsilon_{\mathrm{T}}$", fontsize=16, fontweight="bold")
            if column == 0:
                ax.set_ylabel(r"True Stress, $\sigma_T$ (GPa)", fontsize=16, fontweight="bold")
    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.09, top=0.95, wspace=0.16, hspace=0.20)
    fig.savefig(output_dir / "simulation_all_realizations_2x3.png", dpi=600)
    fig.savefig(output_dir / "simulation_all_realizations_2x3.pdf")
    plt.close(fig)


def write_yield_table(data: dict, output_dir: Path) -> None:
    summary = summarize_yields(data)
    with (output_dir / "yield_strengths.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["element", "orientation", "realization", "yield_strength_GPa", "yield_strain", "group_mean_GPa", "group_min_GPa", "group_max_GPa", "group_std_GPa"]
        )
        for element in ELEMENTS:
            for orientation in ORIENTATIONS:
                stats = summary[(element, orientation)]
                for rep in EXPECTED_REPS[(element, orientation)]:
                    strain, strength = yield_point(data[(element, orientation, rep)])
                    writer.writerow(
                        [element, DISPLAY_ORIENTATION[orientation], rep, f"{strength:.8f}", f"{strain:.8f}", f"{stats['mean']:.8f}", f"{stats['min']:.8f}", f"{stats['max']:.8f}", f"{stats['std']:.8f}"]
                    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=Path("figure_recreation/data"))
    parser.add_argument("--output-dir", type=Path, default=Path("figure_recreation/output"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = load_all(args.base)
    create_summary(data, args.output_dir)
    create_realizations(data, args.output_dir)
    write_yield_table(data, args.output_dir)
    print(f"Wrote figures and yield table to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
