# Prophage-Biofilm MIC-Matrix Thresholds

This repository contains the Python code used to reproduce the simulations,
summary values, and figures for the manuscript:

**Thresholds for Prophage-Mediated Biofilm Persistence in Reduced ODE and PDE
Models of Food-Surface Cleaning**

The model asks when a lysogen with a lower MIC-like susceptibility scale can
nevertheless survive better in a biofilm because it produces more protective
matrix. The code reproduces the analytical-threshold plots, repeated-cleaning
ODE simulations, survival-ratio maps, global sensitivity analysis, mixed-culture
analysis, and one-dimensional PDE penetration demonstration.

## Repository Contents

```text
.
├── README.md
├── requirements.txt
├── reproduce_all.sh
├── scripts/
│   ├── generate_figures.py
│   └── smoke_test.py
├── docs/
│   ├── figure_map.md
│   └── variable_dictionary.md
├── figures/
│   ├── fig1_threshold_phase.pdf
│   ├── fig1_threshold_phase.png
│   └── ... generated manuscript figures
└── results/
    └── summary.txt
```

`figures/` and `results/summary.txt` are included as expected outputs. Running
the script overwrites them with newly generated files.

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 scripts/smoke_test.py
python3 scripts/generate_figures.py
```

Alternatively:

```bash
bash reproduce_all.sh
```

The full figure-generation script may take several minutes depending on the
machine. It writes PDFs and PNGs to `figures/` and numerical checks to
`results/summary.txt`.

## Tested Environment

The package was prepared and tested with:

- Python 3.13.5
- NumPy 2.1.3
- SciPy 1.15.3
- Matplotlib 3.10.0

The `requirements.txt` file uses lower bounds so newer compatible versions can
be used. For exact archival reproduction, pin these versions after creating a
Zenodo release.

## Main Commands

Fast analytical smoke test:

```bash
python3 scripts/smoke_test.py
```

Regenerate all figures and summary values:

```bash
python3 scripts/generate_figures.py
```

## Variable Naming

The script uses manuscript symbols where that improves traceability: `R_M`,
`R_rho`, `B`, `E`, `A`, `S`, and `L` match the equations and figure labels.
Global configuration fields use descriptive names in the `ModelParameters`
dataclass, for example `growth_rate`, `max_killing_rate`,
`matrix_protection_strength`, and `cleaning_duration_days`.

See [`docs/variable_dictionary.md`](docs/variable_dictionary.md) for the full
map between code names, manuscript notation, meanings, and baseline values.

## Figure Map

See [`docs/figure_map.md`](docs/figure_map.md) for the mapping between
manuscript figures, output files, and code routines.

## Key Numerical Checks

Current `results/summary.txt` includes these checks:

- Analytical threshold with `R_M=0.5`, `theta=5`: `R_rho*=1.4` at `a=1`.
- Analytical threshold with `R_M=0.5`, `theta=5`: `R_rho*=1.05714` at `a=10`.
- Baseline repeated-cleaning final biomasses:
  `B_parent(T)=0.0142784`, `B_lysogen(T)=0.0221559`.
- Baseline survival ratio: `B_lysogen(T)/B_parent(T)=1.5517`.
- PDE penetration final mean-exposure ratio: `0.877304`.

## Reproducibility Notes

- The global sensitivity analysis uses deterministic Latin-hypercube sampling
  with seed `20260717`.
- Grid-based ODE calculations use vectorized fourth-order Runge-Kutta steps.
- Single-trajectory monoculture simulations use `scipy.integrate.solve_ivp`.
- The PDE penetration example is a fixed-matrix transport demonstration, not a
  fitted spatial biofilm model.

## Suggested Citation



## License
