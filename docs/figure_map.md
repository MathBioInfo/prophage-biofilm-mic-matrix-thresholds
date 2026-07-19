# Figure Map

Run `python3 scripts/generate_figures.py` from the repository root to regenerate
all files below. PDFs are manuscript-ready vector figures; PNGs are included for
quick inspection on GitHub.

| Manuscript figure | Main output file | Code routine |
| --- | --- | --- |
| Fig. 1 | `figures/fig1_threshold_phase.pdf` | `make_phase_figure` |
| Fig. 2 | `figures/fig2_repeated_cleaning.pdf` | `make_simulation_figure` |
| Fig. 3 | `figures/fig3_survival_ratio_map.pdf` | `make_survival_map` |
| Fig. 4 | `figures/fig4_exposure_threshold_curves.pdf` | `make_exposure_threshold_figure` |
| Fig. 5 | `figures/fig5_cleaning_control_map.pdf` | `make_cleaning_control_map` |
| Fig. 6 | `figures/fig6_global_sensitivity.pdf` | `make_global_sensitivity_figure` |
| Fig. 7 | `figures/fig7_time_to_control.pdf` | `make_time_to_control_figure` |
| Fig. 8 | `figures/fig8_boundary_shift.pdf` | `make_boundary_shift_figure` |
| Fig. 9 | `figures/fig9_pulse_map.pdf` | `make_pulse_map_figure` |
| Fig. 10 | `figures/fig10_initial_maturity.pdf` | `make_initial_maturity_figure` |
| Fig. 11 | `figures/fig11_mixed_culture.pdf` | `make_mixed_culture_figure` |
| Fig. 12 | `figures/fig12_pde_penetration.pdf` | `make_pde_penetration_figure` |

Additional single-panel PDF/PNG files are also generated for inspection and
for possible journal layout changes. Numerical checks are written to
`results/summary.txt`.
