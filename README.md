# Orientation-dependent deformation of noble-metal nanoparticles

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
python scripts/plot_from_csv.py --data data/processed_stress_strain --output reproduced_plot.png
```

The repository distributes CSVs as the archival, software-neutral data format. `scripts/recreate_figures.py` documents the original pickle-based workflow, while `scripts/export_processed_curves.py` shows how the CSVs were produced.

## Analysis definitions

- True strain: `ln(L0/L)`, where `L` is the instantaneous wall separation.
- True stress: the mean of the absolute top- and bottom-wall forces divided by their independently calculated contact areas, converted from eV/Å³ to GPa.
- Simulation yield strength: the true stress immediately before the first stress drop; in the archived calculation this corresponds to the maximum of each processed curve.
- Error-bar extent: minimum to maximum yield strength among independent realizations.
- Experimental CRSS: experimental yield strength multiplied by the Schmid factor of the operative mechanism (`m = 0.2722` for [111] slip; `m = 0.4714` for [001] twinning).

## Notes

The workbook identifies current manuscript-draft values separately from simulation-derived values. No uncertainty has been invented where the draft does not report one.
