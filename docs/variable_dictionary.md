# Variable Dictionary

This repository uses two naming styles deliberately. Variables that appear
directly in equations often keep the manuscript symbols so that code, figures,
and equations can be checked side by side. Configuration parameters use
descriptive field names in `ModelParameters`.

## State Variables

| Code name | Manuscript symbol | Meaning |
| --- | --- | --- |
| `B` | `B` or monoculture biomass | viable biofilm biomass in monoculture simulations |
| `S` | `S` | parental or non-lysogenic biomass in mixed culture |
| `L` | `L` | lysogenic biomass in mixed culture |
| `E` | `E` or `e` | matrix-mediated protection variable |
| `A` | `A` or `a` | sanitizer or antimicrobial exposure |
| `n` | `n` | nutrient variable in the PDE extension |

## Main Dimensionless Ratios

| Code name | Manuscript symbol | Meaning |
| --- | --- | --- |
| `R_M` | `R_M = M_L/M_S` | lysogen-to-parent MIC-like susceptibility ratio |
| `R_rho` | `R_\rho = \rho_L/\rho_S` | lysogen-to-parent matrix-production ratio |
| `theta` | `\theta` or `\Theta/b` | matrix-protection strength |
| `a` | `a = A/M_S` | dimensionless exposure in threshold calculations |

## `ModelParameters` Fields

| Field | Manuscript notation | Meaning | Baseline |
| --- | --- | --- | --- |
| `growth_rate` | `\mu` | maximum net biofilm growth rate | `1.25` day^-1 |
| `max_killing_rate` | `k_max` | maximum sanitizer killing rate | `18.0` day^-1 |
| `matrix_protection_strength` | `\theta` | strength of matrix-mediated protection | `5.0` |
| `matrix_relaxation_rate` | `\epsilon_E` | relaxation rate toward biomass-scaled matrix target | `4.0` day^-1 |
| `matrix_background_loss_rate` | `d_{E0}` | background matrix loss | `0.15` day^-1 |
| `cleaning_matrix_removal_rate` | `\delta_E` | extra matrix removal during cleaning | `1.0` day^-1 |
| `cleaning_sanitizer_input_rate` | `u_0` | sanitizer input during cleaning | `220.0` day^-1 |
| `sanitizer_washout_rate` | `\lambda_A` | sanitizer washout or decay | `18.0` day^-1 |
| `matrix_sanitizer_loss_rate` | `\eta_A` | matrix-associated sanitizer loss | `1.8` |
| `lysogen_loss_rate` | `\alpha` | lysogen fitness or induction cost | `0.015` day^-1 |
| `cleaning_start_day` | -- | first cleaning time | `2.0` days |
| `cleaning_period_days` | -- | interval between cleaning pulses | `1.0` day |
| `cleaning_duration_days` | -- | duration of each cleaning pulse | `2/24` day |
| `final_time_days` | `T` | simulation endpoint | `9.0` days |

## Naming Notes

- `R_M` and `R_rho` are kept in function arguments because these are the
  principal axes of several manuscript figures.
- Grid variables such as `RM`, `RR`, `DUR`, and `DEL` are meshgrid arrays for
  MIC ratio, matrix ratio, cleaning duration, and matrix removal.
- Local variables `Bp`, `Ep`, `Ap`, `Bl`, `El`, and `Al` mean parental and
  lysogenic values of `B`, `E`, and `A` in paired grid simulations.
