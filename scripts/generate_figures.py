#!/usr/bin/env python3
"""Generate reproducible calculations and figures for the prophage-biofilm model.

Variable naming policy
----------------------
The ODE/PDE state variables and dimensionless ratios intentionally keep the
manuscript symbols where that helps verification: B is viable biomass, E is
matrix protection, A is sanitizer exposure, R_M is the lysogen-to-parent MIC
ratio, and R_rho is the lysogen-to-parent matrix-production ratio.

Global model parameters use descriptive dataclass field names such as
growth_rate, max_killing_rate, matrix_protection_strength, and
cleaning_duration_days. See docs/variable_dictionary.md for a complete map.

The calculations are scenario analyses used to test the analytical threshold;
they are not fitted parameter estimates for SapYZUs631.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import LogLocator, NullFormatter

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"
RES_DIR = ROOT / "results"
FIG_DIR.mkdir(parents=True, exist_ok=True)
RES_DIR.mkdir(parents=True, exist_ok=True)

TEXTWIDTH_BP = 468.0
TARGET_AXIS_LABEL_SIZE = 7.5
TARGET_TICK_LABEL_SIZE = 6.75
TARGET_LEGEND_FONT_SIZE = 6.75
TARGET_ANNOTATION_SIZE = 6.75
TARGET_PANEL_LABEL_SIZE = 8.75

AXIS_LABEL_SIZE = 10
TICK_LABEL_SIZE = 9
LEGEND_FONT_SIZE = 9
ANNOTATION_SIZE = 9
PANEL_LABEL_SIZE = 11

plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": TICK_LABEL_SIZE,
    "axes.titlesize": AXIS_LABEL_SIZE,
    "axes.labelsize": AXIS_LABEL_SIZE,
    "xtick.labelsize": TICK_LABEL_SIZE,
    "ytick.labelsize": TICK_LABEL_SIZE,
    "legend.fontsize": LEGEND_FONT_SIZE,
    "figure.titlesize": AXIS_LABEL_SIZE,
    "axes.linewidth": 1.0,
    "lines.linewidth": 2.3,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

PARENT = "#8a5a2b"
LYSOGEN = "#1b7f6a"
BLUE_SHADE = "#d9e8f6"
PHENOTYPE = "#6b2c90"
CONTOUR = "#111111"


def add_panel_labels(axes, labels, x: float = -0.12, y: float = 1.035) -> None:
    for ax, label in zip(axes, labels):
        ax.text(
            x,
            y,
            label,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=PANEL_LABEL_SIZE,
            fontweight="bold",
            color="black",
            clip_on=False,
            zorder=20,
        )


def manuscript_scale(fig, latex_width_fraction: float) -> float:
    source_width_bp = fig.get_size_inches()[0] * 72.0
    return latex_width_fraction * TEXTWIDTH_BP / source_width_bp


def apply_manuscript_typography(fig, latex_width_fraction: float) -> None:
    scale = manuscript_scale(fig, latex_width_fraction)
    axis_size = TARGET_AXIS_LABEL_SIZE / scale
    tick_size = TARGET_TICK_LABEL_SIZE / scale
    legend_size = TARGET_LEGEND_FONT_SIZE / scale
    annotation_size = TARGET_ANNOTATION_SIZE / scale
    panel_size = TARGET_PANEL_LABEL_SIZE / scale
    panel_labels = {"A)", "B)", "C)", "D)"}

    for ax in fig.axes:
        ax.xaxis.label.set_size(axis_size)
        ax.yaxis.label.set_size(axis_size)
        ax.tick_params(axis="both", which="both", labelsize=tick_size)
        ax.xaxis.get_offset_text().set_fontsize(tick_size)
        ax.yaxis.get_offset_text().set_fontsize(tick_size)
        legend = ax.get_legend()
        if legend is not None:
            for label in legend.get_texts():
                label.set_fontsize(legend_size)
        for item in ax.texts:
            item.set_fontsize(panel_size if item.get_text() in panel_labels else annotation_size)


@dataclass(frozen=True)
class ModelParameters:
    growth_rate: float = 1.25
    max_killing_rate: float = 18.0
    matrix_protection_strength: float = 5.0
    matrix_relaxation_rate: float = 4.0
    matrix_background_loss_rate: float = 0.15
    cleaning_matrix_removal_rate: float = 1.0
    cleaning_sanitizer_input_rate: float = 220.0
    sanitizer_washout_rate: float = 18.0
    matrix_sanitizer_loss_rate: float = 1.8
    lysogen_loss_rate: float = 0.015
    cleaning_start_day: float = 2.0
    cleaning_period_days: float = 1.0
    cleaning_duration_days: float = 2.0 / 24.0
    final_time_days: float = 9.0


def style_log_x(ax):
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=6))
    ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=12))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.grid(True, which="major", color="#d0d0d0", lw=0.65, alpha=0.85)
    ax.grid(True, which="minor", color="#e8e8e8", lw=0.45, alpha=0.7)


def intrinsic_kill_ratio(a: np.ndarray | float, R_M: np.ndarray | float) -> np.ndarray | float:
    return (1.0 + a) / (R_M + a)


def matrix_protection_ratio(theta: float, R_rho: np.ndarray | float) -> np.ndarray | float:
    return (1.0 + theta * R_rho) / (1.0 + theta)


def threshold_rrho(a: np.ndarray | float, R_M: np.ndarray | float, theta: float) -> np.ndarray | float:
    qM = intrinsic_kill_ratio(a, R_M)
    return (qM * (1.0 + theta) - 1.0) / theta


def is_cleaning(t: float, p: ModelParameters) -> bool:
    if t < p.cleaning_start_day:
        return False
    phase = (t - p.cleaning_start_day) % p.cleaning_period_days
    return phase < p.cleaning_duration_days


def cleaning_windows(p: ModelParameters) -> list[tuple[float, float]]:
    windows = []
    t = p.cleaning_start_day
    while t < p.final_time_days:
        windows.append((t, min(t + p.cleaning_duration_days, p.final_time_days)))
        t += p.cleaning_period_days
    return windows


def simulate_monoculture(R_rho: float, R_M: float, lysogen: bool, p: ModelParameters):
    rho = R_rho if lysogen else 1.0
    M = R_M if lysogen else 1.0
    alpha = p.lysogen_loss_rate if lysogen else 0.0
    y0 = np.array([0.82, 0.82 * rho, 0.0])

    def rhs(t, y):
        B, E, A = np.maximum(y, 0.0)
        clean = is_cleaning(t, p)
        u = p.cleaning_sanitizer_input_rate if clean else 0.0
        delta = p.cleaning_matrix_removal_rate if clean else 0.0
        growth = p.growth_rate * B * max(0.0, 1.0 - B)
        kill = p.max_killing_rate * (A / (A + M + 1e-12)) / (1.0 + p.matrix_protection_strength * E)
        dB = growth - kill * B - alpha * B
        dE = p.matrix_relaxation_rate * (rho * B - E) - p.matrix_background_loss_rate * E - delta * E
        dA = u - p.sanitizer_washout_rate * A - p.matrix_sanitizer_loss_rate * E * A
        return [dB, dE, dA]

    t_eval = np.linspace(0.0, p.final_time_days, 1800)
    sol = solve_ivp(rhs, (0.0, p.final_time_days), y0, t_eval=t_eval, max_step=0.004, rtol=1e-8, atol=1e-10)
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol


def make_phase_figure(theta: float = 5.0) -> None:
    Rm = np.logspace(np.log10(1 / 1024), 0, 420)
    Rrho = np.linspace(0.8, 3.2, 320)
    X, Y = np.meshgrid(Rm, Rrho)

    def draw_threshold_axis(ax, a: float, title: str, show_ylabel: bool = True) -> None:
        qM = intrinsic_kill_ratio(a, X)
        qE = matrix_protection_ratio(theta, Y)
        benefit = qE > qM
        ax.contourf(X, Y, benefit.astype(float), levels=[-0.1, 0.5, 1.1], colors=["#f5efe6", "#d8ecd2"])
        thr = threshold_rrho(a, Rm, theta)
        ax.plot(Rm, thr, color="#124f70", lw=2.8, label=r"$R_\rho^*$")
        ax.contour(X, Y, qE - qM, levels=[0.0], colors=CONTOUR, linewidths=1.4)
        ax.fill_betweenx([1.1, 1.4], 1 / 1024, 1 / 2, color=PHENOTYPE, alpha=0.18, label="observed scale")
        style_log_x(ax)
        ax.set_ylim(0.8, 3.2)
        ax.set_xlim(1 / 1024, 1.0)
        ax.set_xlabel(r"MIC ratio, $R_M=M_L/M_S$", fontsize=AXIS_LABEL_SIZE)
        if show_ylabel:
            ax.set_ylabel(r"Matrix ratio, $R_\rho=\rho_L/\rho_S$", fontsize=AXIS_LABEL_SIZE)
        ax.tick_params(axis="both", which="both", labelsize=TICK_LABEL_SIZE)
        ax.text(0.00125, 2.92, "matrix benefit", ha="left", va="top", fontsize=ANNOTATION_SIZE, color="#245c2f")
        ax.text(0.00125, 0.92, "MIC cost", ha="left", va="bottom", fontsize=ANNOTATION_SIZE, color="#725238")

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.3), constrained_layout=True)
    draw_threshold_axis(axes[0], 1.0, r"Exposure near parental MIC ($a=1$)", show_ylabel=True)
    draw_threshold_axis(axes[1], 10.0, r"High exposure ($a=10$)", show_ylabel=False)
    add_panel_labels(axes, ["A)", "B)"])
    axes[1].legend(loc="upper right", frameon=True, framealpha=0.95, fontsize=LEGEND_FONT_SIZE)
    apply_manuscript_typography(fig, 0.98)
    fig.savefig(FIG_DIR / "fig1_threshold_phase.pdf")
    fig.savefig(FIG_DIR / "fig1_threshold_phase.png")
    plt.close(fig)

    for a, title, stem in [
        (1.0, r"Exposure near parental MIC ($a=1$)", "fig1a_threshold_near_mic"),
        (10.0, r"High exposure ($a=10$)", "fig1b_threshold_high_exposure"),
    ]:
        single_fig, single_ax = plt.subplots(figsize=(7.2, 5.2), constrained_layout=True)
        draw_threshold_axis(single_ax, a, title, show_ylabel=True)
        single_ax.legend(loc="upper right", frameon=True, framealpha=0.95, fontsize=LEGEND_FONT_SIZE)
        single_fig.savefig(FIG_DIR / f"{stem}.pdf")
        single_fig.savefig(FIG_DIR / f"{stem}.png")
        plt.close(single_fig)


def make_simulation_figure() -> dict[str, float]:
    p = ModelParameters()
    R_M = 0.50
    R_rho = 1.40
    parent = simulate_monoculture(R_rho=R_rho, R_M=R_M, lysogen=False, p=p)
    lysogen = simulate_monoculture(R_rho=R_rho, R_M=R_M, lysogen=True, p=p)

    fig, axes = plt.subplots(3, 1, figsize=(7.2, 5.2), sharex=True, constrained_layout=True)
    for ax in axes:
        for lo, hi in cleaning_windows(p):
            ax.axvspan(lo, hi, color=BLUE_SHADE, alpha=0.85, lw=0)
        ax.grid(True, color="#d7d7d7", lw=0.65, alpha=0.85)

    axes[0].plot(parent.t, parent.y[0], color=PARENT, lw=2.7, label="Parental")
    axes[0].plot(lysogen.t, lysogen.y[0], color=LYSOGEN, lw=2.7, label="Lysogenic")
    axes[0].set_yscale("log")
    axes[0].set_ylim(8e-3, 1.1)
    axes[0].set_ylabel(r"Viable biomass, $B$")
    axes[0].legend(loc="lower left", frameon=True, framealpha=0.95, ncol=2)

    axes[1].plot(parent.t, parent.y[1], color=PARENT, lw=2.7)
    axes[1].plot(lysogen.t, lysogen.y[1], color=LYSOGEN, lw=2.7)
    axes[1].set_ylabel(r"Matrix, $E$")

    axes[2].plot(parent.t, parent.y[2], color=PARENT, lw=2.2)
    axes[2].plot(lysogen.t, lysogen.y[2], color=LYSOGEN, lw=2.2)
    axes[2].set_ylabel(r"Sanitizer, $A$")
    axes[2].set_xlabel(r"Time (days)")
    apply_manuscript_typography(fig, 0.82)

    fig.savefig(FIG_DIR / "fig2_repeated_cleaning.pdf")
    fig.savefig(FIG_DIR / "fig2_repeated_cleaning.png")
    plt.close(fig)

    return {
        "R_M": R_M,
        "R_rho": R_rho,
        "final_parent_biomass": float(parent.y[0, -1]),
        "final_lysogen_biomass": float(lysogen.y[0, -1]),
        "final_survival_ratio_L_over_S": float(lysogen.y[0, -1] / max(parent.y[0, -1], 1e-12)),
        "max_parent_sanitizer": float(parent.y[2].max()),
        "max_lysogen_sanitizer": float(lysogen.y[2].max()),
    }


def rk4_step(rhs, t: float, y: tuple[np.ndarray, np.ndarray, np.ndarray], dt: float):
    k1 = rhs(t, y)
    y2 = tuple(yi + 0.5 * dt * ki for yi, ki in zip(y, k1))
    k2 = rhs(t + 0.5 * dt, y2)
    y3 = tuple(yi + 0.5 * dt * ki for yi, ki in zip(y, k2))
    k3 = rhs(t + 0.5 * dt, y3)
    y4 = tuple(yi + dt * ki for yi, ki in zip(y, k3))
    k4 = rhs(t + dt, y4)
    return tuple(np.maximum(yi + dt * (ki1 + 2 * ki2 + 2 * ki3 + ki4) / 6.0, 0.0)
                 for yi, ki1, ki2, ki3, ki4 in zip(y, k1, k2, k3, k4))


def integrate_monoculture_grid(
    R_rho: np.ndarray,
    R_M: np.ndarray,
    lysogen: bool,
    p: ModelParameters,
    dt: float = 0.003,
    duration_grid: np.ndarray | None = None,
    delta_grid: np.ndarray | None = None,
    initial_B: np.ndarray | float | None = None,
    initial_E: np.ndarray | float | None = None,
    initial_A: np.ndarray | float | None = None,
):
    rho = R_rho if lysogen else np.ones_like(R_rho)
    M = R_M if lysogen else np.ones_like(R_M)
    alpha = p.lysogen_loss_rate if lysogen else 0.0
    duration = p.cleaning_duration_days if duration_grid is None else duration_grid
    delta_clean = p.cleaning_matrix_removal_rate if delta_grid is None else delta_grid
    B = np.full_like(R_rho, 0.82, dtype=float) if initial_B is None else np.asarray(initial_B, dtype=float) + np.zeros_like(R_rho, dtype=float)
    E = 0.82 * rho if initial_E is None else np.asarray(initial_E, dtype=float) + np.zeros_like(R_rho, dtype=float)
    A = np.zeros_like(R_rho, dtype=float) if initial_A is None else np.asarray(initial_A, dtype=float) + np.zeros_like(R_rho, dtype=float)

    def rhs(t, y):
        Bv, Ev, Av = (np.maximum(v, 0.0) for v in y)
        if t < p.cleaning_start_day:
            clean = np.zeros_like(Bv, dtype=bool)
        else:
            phase = (t - p.cleaning_start_day) % p.cleaning_period_days
            clean = phase < duration
        clean_float = np.asarray(clean, dtype=float)
        u = p.cleaning_sanitizer_input_rate * clean_float
        delta = delta_clean * clean_float
        growth = p.growth_rate * Bv * np.maximum(0.0, 1.0 - Bv)
        kill = p.max_killing_rate * (Av / (Av + M + 1e-12)) / (1.0 + p.matrix_protection_strength * Ev)
        dB = growth - kill * Bv - alpha * Bv
        dE = p.matrix_relaxation_rate * (rho * Bv - Ev) - p.matrix_background_loss_rate * Ev - delta * Ev
        dA = u - p.sanitizer_washout_rate * Av - p.matrix_sanitizer_loss_rate * Ev * Av
        return dB, dE, dA

    t = 0.0
    while t < p.final_time_days - 1e-12:
        step = min(dt, p.final_time_days - t)
        B, E, A = rk4_step(rhs, t, (B, E, A), step)
        t += step
    return B, E, A


def make_survival_map() -> dict[str, float]:
    p = ModelParameters()
    rm = np.logspace(np.log10(1 / 1024), 0.0, 210)
    rr = np.linspace(1.0, 2.8, 170)
    RM, RR = np.meshgrid(rm, rr)

    parent_B, _, _ = integrate_monoculture_grid(np.ones_like(RM), np.ones_like(RM), lysogen=False, p=p)
    lys_B, _, _ = integrate_monoculture_grid(RR, RM, lysogen=True, p=p)
    ratio = lys_B / np.maximum(parent_B, 1e-12)
    log_ratio = np.log10(np.maximum(ratio, 1e-12))

    fig, ax = plt.subplots(figsize=(7.2, 5.6), constrained_layout=True)
    levels = np.linspace(-2.0, 2.0, 41)
    cf = ax.contourf(
        RM,
        RR,
        np.clip(log_ratio, -2.0, 2.0),
        levels=levels,
        cmap="BrBG",
        zorder=0,
    )
    ax.fill_betweenx(
        [1.1, 1.4],
        1 / 1024,
        1 / 2,
        color=PHENOTYPE,
        alpha=0.10,
        edgecolor="none",
        linewidth=0,
        zorder=1,
    )
    ax.contour(RM, RR, ratio, levels=[1.0], colors=CONTOUR, linewidths=2.4, zorder=3)
    ax.plot([], [], color=CONTOUR, lw=2.4, label="equal survival")

    ax.set_xscale("log")
    ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=6))
    ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=12))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.grid(False)
    ax.set_axisbelow(False)

    ax.set_xlim(1 / 1024, 1.0)
    ax.set_ylim(1.0, 2.8)
    ax.set_xlabel(r"MIC ratio, $R_M=M_L/M_S$", color="black")
    ax.set_ylabel(r"Matrix ratio, $R_\rho=\rho_L/\rho_S$", color="black")
    ax.tick_params(axis="both", which="both", colors="black", width=0.9)
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(1.0)

    cmap = plt.get_cmap("BrBG")
    legend_handles = [
        Patch(facecolor=cmap(0.82), edgecolor="#4b4b4b", label=r"Lysogen advantage: $B_L(T)/B_S(T)>1$"),
        Patch(facecolor=cmap(0.18), edgecolor="#4b4b4b", label=r"Parental advantage: $B_L(T)/B_S(T)<1$"),
        Line2D([0], [0], color=CONTOUR, lw=2.4, label=r"Equal survival: $B_L(T)/B_S(T)=1$"),
    ]
    legend = ax.legend(
        handles=legend_handles,
        loc="lower left",
        frameon=True,
        framealpha=0.96,
        facecolor="white",
        edgecolor="#cfcfcf",
        handlelength=1.6,
        borderpad=0.65,
        labelspacing=0.45,
    )
    for item in legend.get_texts():
        item.set_color("black")

    cbar = fig.colorbar(cf, ax=ax, pad=0.02, ticks=[-2, -1, 0, 1, 2])
    cbar.set_label(r"$\log_{10}(B_L(T)/B_S(T))$", color="black", fontsize=AXIS_LABEL_SIZE)
    cbar.ax.tick_params(colors="black", width=0.8, labelsize=TICK_LABEL_SIZE)
    cbar.outline.set_edgecolor("black")
    cbar.outline.set_linewidth(0.8)
    apply_manuscript_typography(fig, 0.82)

    fig.savefig(FIG_DIR / "fig3_survival_ratio_map.pdf")
    fig.savefig(FIG_DIR / "fig3_survival_ratio_map.png")
    plt.close(fig)

    return {
        "survival_map_parent_final_biomass": float(parent_B[0, 0]),
        "survival_map_min_ratio": float(np.min(ratio)),
        "survival_map_max_ratio": float(np.max(ratio)),
        "survival_map_fraction_lysogen_advantage": float(np.mean(ratio > 1.0)),
    }


def make_exposure_threshold_figure(theta: float = 5.0) -> dict[str, float]:
    a_values = np.logspace(-1.0, 2.0, 500)
    rm_values = [0.5, 0.2, 0.01, 1 / 1024]
    labels = [r"$R_M=0.5$", r"$R_M=0.2$", r"$R_M=0.01$", r"$R_M=1/1024$"]
    colors = ["#124f70", "#2a7f62", "#b4641d", "#7c3f98"]

    fig, ax = plt.subplots(figsize=(7.2, 5.2), constrained_layout=True)
    for rm, label, color in zip(rm_values, labels, colors):
        ax.plot(a_values, threshold_rrho(a_values, rm, theta), label=label, color=color, lw=2.8)
    ax.axhspan(1.1, 1.4, color=PHENOTYPE, alpha=0.13, label="observed biofilm scale")
    ax.set_xscale("log")
    ax.set_ylim(0.9, 5.0)
    ax.set_xlabel(r"Dimensionless exposure, $a=A/M_S$")
    ax.set_ylabel(r"Threshold matrix ratio, $R_\rho^*$")
    ax.grid(True, which="major", color="#d0d0d0", lw=0.65, alpha=0.85)
    ax.grid(True, which="minor", color="#e8e8e8", lw=0.45, alpha=0.7)
    ax.legend(loc="upper right", frameon=True, framealpha=0.95)
    apply_manuscript_typography(fig, 0.82)
    fig.savefig(FIG_DIR / "fig4_exposure_threshold_curves.pdf")
    fig.savefig(FIG_DIR / "fig4_exposure_threshold_curves.png")
    plt.close(fig)

    return {
        "threshold_RM_0p5_a_0p1": float(threshold_rrho(0.1, 0.5, theta)),
        "threshold_RM_0p5_a_1": float(threshold_rrho(1.0, 0.5, theta)),
        "threshold_RM_0p5_a_10": float(threshold_rrho(10.0, 0.5, theta)),
        "threshold_RM_0p5_a_100": float(threshold_rrho(100.0, 0.5, theta)),
    }


def make_cleaning_control_map() -> dict[str, float]:
    p = ModelParameters()
    R_M = 0.5
    R_rho = 1.4
    durations = np.linspace(0.5 / 24.0, 6.0 / 24.0, 120)
    deltas = np.linspace(0.0, 5.0, 120)
    DUR, DEL = np.meshgrid(durations, deltas)
    RM = np.full_like(DUR, R_M)
    RR = np.full_like(DUR, R_rho)

    parent_B, _, _ = integrate_monoculture_grid(
        np.ones_like(RM), np.ones_like(RM), lysogen=False, p=p, duration_grid=DUR, delta_grid=DEL
    )
    lys_B, _, _ = integrate_monoculture_grid(RR, RM, lysogen=True, p=p, duration_grid=DUR, delta_grid=DEL)
    ratio = lys_B / np.maximum(parent_B, 1e-12)
    log_ratio = np.log10(np.maximum(ratio, 1e-12))

    fig, ax = plt.subplots(figsize=(7.2, 5.2), constrained_layout=True)
    levels = np.linspace(-1.5, 1.5, 37)
    cf = ax.contourf(DUR * 24.0, DEL, np.clip(log_ratio, -1.5, 1.5), levels=levels, cmap="BrBG")
    ax.contour(DUR * 24.0, DEL, ratio, levels=[1.0], colors=CONTOUR, linewidths=2.2)
    ax.scatter([p.cleaning_duration_days * 24.0], [p.cleaning_matrix_removal_rate], s=70, color="#c51b7d", edgecolor="white", linewidth=1.0, zorder=5, label="baseline")
    ax.set_xlabel(r"Cleaning duration (h day$^{-1}$)")
    ax.set_ylabel(r"Matrix removal rate during cleaning, $\delta_E$")
    ax.grid(False)
    ax.tick_params(axis="both", which="both", colors="black", width=0.9)
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(1.0)
    ax.legend(loc="upper right", frameon=True, framealpha=0.95, facecolor="white", edgecolor="#cfcfcf")
    cbar = fig.colorbar(cf, ax=ax, pad=0.02, ticks=[-1.5, -0.75, 0, 0.75, 1.5])
    cbar.set_label(r"$\log_{10}(B_L(T)/B_S(T))$", color="black", fontsize=AXIS_LABEL_SIZE)
    cbar.ax.tick_params(colors="black", width=0.8, labelsize=TICK_LABEL_SIZE)
    cbar.outline.set_edgecolor("black")
    cbar.outline.set_linewidth(0.8)
    apply_manuscript_typography(fig, 0.82)
    fig.savefig(FIG_DIR / "fig5_cleaning_control_map.pdf")
    fig.savefig(FIG_DIR / "fig5_cleaning_control_map.png")
    plt.close(fig)

    baseline_idx = np.unravel_index(np.argmin((DUR - p.cleaning_duration_days) ** 2 + (DEL - p.cleaning_matrix_removal_rate) ** 2), DUR.shape)
    return {
        "cleaning_map_min_ratio": float(np.min(ratio)),
        "cleaning_map_max_ratio": float(np.max(ratio)),
        "cleaning_map_fraction_lysogen_advantage": float(np.mean(ratio > 1.0)),
        "cleaning_map_baseline_ratio_nearest_grid": float(ratio[baseline_idx]),
    }



def latin_hypercube(n_samples: int, n_parameters: int, seed: int = 20260717) -> np.ndarray:
    rng = np.random.default_rng(seed)
    design = np.empty((n_samples, n_parameters), dtype=float)
    base = (np.arange(n_samples) + rng.random(n_samples)) / n_samples
    for j in range(n_parameters):
        design[:, j] = rng.permutation(base)
    return design


def rank_vector(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(x), dtype=float)
    return ranks


def partial_rank_correlations(samples: np.ndarray, response: np.ndarray) -> np.ndarray:
    ranked_x = np.column_stack([rank_vector(samples[:, j]) for j in range(samples.shape[1])])
    ranked_y = rank_vector(response)
    prcc = np.empty(samples.shape[1], dtype=float)
    for j in range(samples.shape[1]):
        others = np.delete(ranked_x, j, axis=1)
        design = np.column_stack([np.ones(samples.shape[0]), others])
        beta_x, *_ = np.linalg.lstsq(design, ranked_x[:, j], rcond=None)
        beta_y, *_ = np.linalg.lstsq(design, ranked_y, rcond=None)
        residual_x = ranked_x[:, j] - design @ beta_x
        residual_y = ranked_y - design @ beta_y
        prcc[j] = np.corrcoef(residual_x, residual_y)[0, 1]
    return prcc


def make_global_sensitivity_figure() -> dict[str, float]:
    n_samples = 700
    unit = latin_hypercube(n_samples, 11)
    names = [
        r"MIC ratio $R_M$",
        r"matrix ratio $R_\rho$",
        r"protection $\theta$",
        r"matrix relaxation $\epsilon_E$",
        "cleaning duration",
        r"matrix removal $\delta_E$",
        r"sanitizer input $u_0$",
        r"washout $\lambda_A$",
        r"lysogen cost $\alpha$",
        r"max killing $k_{\max}$",
        r"matrix loss $\eta_A$",
    ]
    values = {
        "R_M": 10 ** (np.log10(1 / 1024) + unit[:, 0] * (0.0 - np.log10(1 / 1024))),
        "R_rho": 1.0 + unit[:, 1] * 1.8,
        "theta": 1.0 + unit[:, 2] * 9.0,
        "eps": 10 ** (np.log10(0.5) + unit[:, 3] * (np.log10(8.0) - np.log10(0.5))),
        "duration": (0.5 + unit[:, 4] * 5.5) / 24.0,
        "delta_clean": unit[:, 5] * 5.0,
        "u0": 80.0 + unit[:, 6] * 320.0,
        "lamA": 8.0 + unit[:, 7] * 27.0,
        "alphaL": unit[:, 8] * 0.05,
        "kmax": 8.0 + unit[:, 9] * 24.0,
        "etaA": 0.5 + unit[:, 10] * 3.5,
    }
    p = replace(
        ModelParameters(),
        matrix_protection_strength=values["theta"],
        matrix_relaxation_rate=values["eps"],
        cleaning_sanitizer_input_rate=values["u0"],
        sanitizer_washout_rate=values["lamA"],
        lysogen_loss_rate=values["alphaL"],
        max_killing_rate=values["kmax"],
        matrix_sanitizer_loss_rate=values["etaA"],
    )
    RM = values["R_M"]
    RR = values["R_rho"]
    parent_B, _, _ = integrate_monoculture_grid(
        np.ones_like(RM),
        np.ones_like(RM),
        lysogen=False,
        p=p,
        duration_grid=values["duration"],
        delta_grid=values["delta_clean"],
    )
    lys_B, _, _ = integrate_monoculture_grid(
        RR,
        RM,
        lysogen=True,
        p=p,
        duration_grid=values["duration"],
        delta_grid=values["delta_clean"],
    )
    response = np.log10(np.maximum(lys_B / np.maximum(parent_B, 1e-12), 1e-12))
    sample_matrix = np.column_stack([
        np.log10(values["R_M"]),
        values["R_rho"],
        values["theta"],
        np.log10(values["eps"]),
        values["duration"] * 24.0,
        values["delta_clean"],
        values["u0"],
        values["lamA"],
        values["alphaL"],
        values["kmax"],
        values["etaA"],
    ])
    prcc = partial_rank_correlations(sample_matrix, response)
    order = np.argsort(np.abs(prcc))
    ordered_names = [names[i] for i in order]
    ordered_values = prcc[order]
    colors = np.where(ordered_values >= 0, LYSOGEN, PARENT)

    fig, ax = plt.subplots(figsize=(7.2, 5.6), constrained_layout=True)
    y = np.arange(len(ordered_names))
    ax.barh(y, ordered_values, color=colors, edgecolor="black", linewidth=0.45)
    ax.axvline(0.0, color="black", lw=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels(ordered_names)
    ax.set_xlim(-1.0, 1.0)
    ax.set_xlabel(r"Partial rank correlation with $\log_{10}(B_L(T)/B_S(T))$")
    ax.grid(False)
    ax.tick_params(axis="both", which="both", colors="black")
    for spine in ax.spines.values():
        spine.set_color("black")
    ax.text(0.02, 0.03, "negative: favors parental survival", transform=ax.transAxes, ha="left", va="bottom", fontsize=ANNOTATION_SIZE, color="black")
    ax.text(0.98, 0.03, "positive: favors lysogen survival", transform=ax.transAxes, ha="right", va="bottom", fontsize=ANNOTATION_SIZE, color="black")
    apply_manuscript_typography(fig, 0.82)
    fig.savefig(FIG_DIR / "fig6_global_sensitivity.pdf")
    fig.savefig(FIG_DIR / "fig6_global_sensitivity.png")
    plt.close(fig)

    strongest = int(np.argmax(np.abs(prcc)))
    return {
        "sensitivity_samples": float(n_samples),
        "sensitivity_response_min": float(np.min(response)),
        "sensitivity_response_max": float(np.max(response)),
        "sensitivity_fraction_lysogen_advantage": float(np.mean(response > 0.0)),
        "sensitivity_strongest_signed_prcc": float(prcc[strongest]),
        "sensitivity_strongest_abs_prcc": float(abs(prcc[strongest])),
        "sensitivity_prcc_log_R_M": float(prcc[0]),
        "sensitivity_prcc_R_rho": float(prcc[1]),
        "sensitivity_prcc_theta": float(prcc[2]),
        "sensitivity_prcc_log_epsilon_E": float(prcc[3]),
        "sensitivity_prcc_cleaning_duration": float(prcc[4]),
        "sensitivity_prcc_delta_E": float(prcc[5]),
        "sensitivity_prcc_u0": float(prcc[6]),
        "sensitivity_prcc_lambda_A": float(prcc[7]),
        "sensitivity_prcc_alpha": float(prcc[8]),
        "sensitivity_prcc_kmax": float(prcc[9]),
        "sensitivity_prcc_eta_A": float(prcc[10]),
    }


def make_time_to_control_figure() -> dict[str, float]:
    p = ModelParameters()
    R_M = 0.5
    R_rho = 1.4
    threshold = 0.02
    durations = np.linspace(0.5 / 24.0, 6.0 / 24.0, 130)
    deltas = np.linspace(0.0, 5.0, 130)
    DUR, DEL = np.meshgrid(durations, deltas)
    shape = DUR.shape

    Bp = np.full(shape, 0.82, dtype=float)
    Ep = np.full(shape, 0.82, dtype=float)
    Ap = np.zeros(shape, dtype=float)
    Bl = np.full(shape, 0.82, dtype=float)
    El = np.full(shape, 0.82 * R_rho, dtype=float)
    Al = np.zeros(shape, dtype=float)
    control_time = np.full(shape, np.nan, dtype=float)

    def rhs_parent(t, y):
        Bv, Ev, Av = (np.maximum(v, 0.0) for v in y)
        if t < p.cleaning_start_day:
            clean = np.zeros_like(Bv, dtype=bool)
        else:
            clean = ((t - p.cleaning_start_day) % p.cleaning_period_days) < DUR
        clean_float = np.asarray(clean, dtype=float)
        u = p.cleaning_sanitizer_input_rate * clean_float
        delta = DEL * clean_float
        growth = p.growth_rate * Bv * np.maximum(0.0, 1.0 - Bv)
        kill = p.max_killing_rate * (Av / (Av + 1.0 + 1e-12)) / (1.0 + p.matrix_protection_strength * Ev)
        dB = growth - kill * Bv
        dE = p.matrix_relaxation_rate * (Bv - Ev) - p.matrix_background_loss_rate * Ev - delta * Ev
        dA = u - p.sanitizer_washout_rate * Av - p.matrix_sanitizer_loss_rate * Ev * Av
        return dB, dE, dA

    def rhs_lysogen(t, y):
        Bv, Ev, Av = (np.maximum(v, 0.0) for v in y)
        if t < p.cleaning_start_day:
            clean = np.zeros_like(Bv, dtype=bool)
        else:
            clean = ((t - p.cleaning_start_day) % p.cleaning_period_days) < DUR
        clean_float = np.asarray(clean, dtype=float)
        u = p.cleaning_sanitizer_input_rate * clean_float
        delta = DEL * clean_float
        growth = p.growth_rate * Bv * np.maximum(0.0, 1.0 - Bv)
        kill = p.max_killing_rate * (Av / (Av + R_M + 1e-12)) / (1.0 + p.matrix_protection_strength * Ev)
        dB = growth - kill * Bv - p.lysogen_loss_rate * Bv
        dE = p.matrix_relaxation_rate * (R_rho * Bv - Ev) - p.matrix_background_loss_rate * Ev - delta * Ev
        dA = u - p.sanitizer_washout_rate * Av - p.matrix_sanitizer_loss_rate * Ev * Av
        return dB, dE, dA

    t = 0.0
    dt = 0.003
    while t < p.final_time_days - 1e-12:
        step = min(dt, p.final_time_days - t)
        Bp, Ep, Ap = rk4_step(rhs_parent, t, (Bp, Ep, Ap), step)
        Bl, El, Al = rk4_step(rhs_lysogen, t, (Bl, El, Al), step)
        t += step
        newly_controlled = np.isnan(control_time) & (np.maximum(Bp, Bl) <= threshold)
        control_time[newly_controlled] = t

    controlled = np.isfinite(control_time)
    masked_time = np.ma.masked_invalid(control_time)
    cmap = plt.get_cmap("viridis_r").copy()
    cmap.set_bad("#e6e6e6")

    fig, ax = plt.subplots(figsize=(7.2, 5.2), constrained_layout=True)
    levels = np.linspace(p.cleaning_start_day, p.final_time_days, 29)
    cf = ax.contourf(DUR * 24.0, DEL, masked_time, levels=levels, cmap=cmap)
    ax.contour(DUR * 24.0, DEL, controlled.astype(float), levels=[0.5], colors=CONTOUR, linewidths=2.1)
    ax.scatter([p.cleaning_duration_days * 24.0], [p.cleaning_matrix_removal_rate], s=70, color="#c51b7d", edgecolor="white", linewidth=1.0, zorder=5, label="baseline")
    ax.plot([], [], color=CONTOUR, lw=2.1, label="control boundary")
    ax.set_xlabel(r"Cleaning duration (h day$^{-1}$)")
    ax.set_ylabel(r"Matrix removal rate during cleaning, $\delta_E$")
    ax.grid(False)
    ax.tick_params(axis="both", which="both", colors="black", width=0.9)
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(1.0)
    ax.legend(loc="upper right", frameon=True, framealpha=0.95, facecolor="white", edgecolor="#cfcfcf")
    cbar = fig.colorbar(cf, ax=ax, pad=0.02)
    cbar.set_label("First threshold crossing time (days)", color="black", fontsize=AXIS_LABEL_SIZE)
    cbar.ax.tick_params(colors="black", width=0.8, labelsize=TICK_LABEL_SIZE)
    cbar.outline.set_edgecolor("black")
    cbar.outline.set_linewidth(0.8)
    apply_manuscript_typography(fig, 0.82)
    fig.savefig(FIG_DIR / "fig7_time_to_control.pdf")
    fig.savefig(FIG_DIR / "fig7_time_to_control.png")
    plt.close(fig)

    baseline_idx = np.unravel_index(np.argmin((DUR - p.cleaning_duration_days) ** 2 + (DEL - p.cleaning_matrix_removal_rate) ** 2), DUR.shape)
    finite_times = control_time[controlled]
    return {
        "time_control_threshold": threshold,
        "time_control_fraction_reached": float(np.mean(controlled)),
        "time_control_min_days": float(np.min(finite_times)) if finite_times.size else float("nan"),
        "time_control_median_days": float(np.median(finite_times)) if finite_times.size else float("nan"),
        "time_control_baseline_days": float(control_time[baseline_idx]) if controlled[baseline_idx] else float("nan"),
    }


def boundary_from_ratio(RM: np.ndarray, RR: np.ndarray, ratio: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rm_vals = RM[0, :]
    rr_vals = RR[:, 0]
    boundary = np.full_like(rm_vals, np.nan, dtype=float)
    for j in range(len(rm_vals)):
        y = ratio[:, j] - 1.0
        signs = np.where(np.signbit(y[:-1]) != np.signbit(y[1:]))[0]
        if signs.size:
            i = signs[0]
            y0, y1 = y[i], y[i + 1]
            if abs(y1 - y0) > 1e-14:
                boundary[j] = rr_vals[i] - y0 * (rr_vals[i + 1] - rr_vals[i]) / (y1 - y0)
            else:
                boundary[j] = rr_vals[i]
    valid = np.isfinite(boundary)
    return rm_vals[valid], boundary[valid]


def make_boundary_shift_figure() -> dict[str, float]:
    p0 = ModelParameters()
    rm = np.logspace(np.log10(1 / 1024), 0.0, 120)
    rr = np.linspace(1.0, 2.8, 100)
    RM, RR = np.meshgrid(rm, rr)
    durations_h = [1.0, 2.0, 4.0, 6.0]
    colors = ["#124f70", "#2a7f62", "#b4641d", "#7c3f98"]

    fig, ax = plt.subplots(figsize=(7.2, 5.2), constrained_layout=True)
    summary: dict[str, float] = {}
    for duration_h, color in zip(durations_h, colors):
        duration = np.full_like(RM, duration_h / 24.0)
        parent_B, _, _ = integrate_monoculture_grid(np.ones_like(RM), np.ones_like(RM), False, p0, dt=0.004, duration_grid=duration)
        lys_B, _, _ = integrate_monoculture_grid(RR, RM, True, p0, dt=0.004, duration_grid=duration)
        ratio = lys_B / np.maximum(parent_B, 1e-12)
        x, y = boundary_from_ratio(RM, RR, ratio)
        if len(x):
            ax.plot(x, y, color=color, lw=2.7, label=f"{duration_h:g} h/day")
        idx = np.argmin(np.abs(rm - 0.5))
        col = ratio[:, idx] - 1.0
        cross = np.where(np.signbit(col[:-1]) != np.signbit(col[1:]))[0]
        key = f"boundary_Rrho_at_RM_0p5_duration_{duration_h:g}h".replace(".", "p")
        if cross.size:
            i = cross[0]
            y0, y1 = col[i], col[i + 1]
            summary[key] = float(rr[i] - y0 * (rr[i + 1] - rr[i]) / (y1 - y0))
        else:
            summary[key] = float("nan")

    ax.fill_betweenx([1.1, 1.4], 1 / 1024, 1 / 2, color=PHENOTYPE, alpha=0.10, edgecolor="none")
    style_log_x(ax)
    ax.grid(False)
    ax.set_xlim(1 / 1024, 1.0)
    ax.set_ylim(1.0, 2.8)
    ax.set_xlabel(r"MIC ratio, $R_M=M_L/M_S$")
    ax.set_ylabel(r"Equal-survival matrix ratio, $R_\rho$")
    ax.legend(loc="upper right", frameon=True, framealpha=0.96, facecolor="white", edgecolor="#cfcfcf")
    for spine in ax.spines.values():
        spine.set_color("black")
    apply_manuscript_typography(fig, 0.82)
    fig.savefig(FIG_DIR / "fig8_boundary_shift.pdf")
    fig.savefig(FIG_DIR / "fig8_boundary_shift.png")
    plt.close(fig)
    return summary


def daily_map_B(B0: np.ndarray, rho: float, M: float, alpha: float, p: ModelParameters, dt: float = 0.0015) -> np.ndarray:
    B = np.asarray(B0, dtype=float).copy()
    E = rho * B
    A = np.zeros_like(B)

    def rhs(t, y):
        Bv, Ev, Av = (np.maximum(v, 0.0) for v in y)
        clean = t < p.cleaning_duration_days
        clean_float = 1.0 if clean else 0.0
        growth = p.growth_rate * Bv * np.maximum(0.0, 1.0 - Bv)
        kill = p.max_killing_rate * (Av / (Av + M + 1e-12)) / (1.0 + p.matrix_protection_strength * Ev)
        dB = growth - kill * Bv - alpha * Bv
        dE = p.matrix_relaxation_rate * (rho * Bv - Ev) - p.matrix_background_loss_rate * Ev - p.cleaning_matrix_removal_rate * clean_float * Ev
        dA = p.cleaning_sanitizer_input_rate * clean_float - p.sanitizer_washout_rate * Av - p.matrix_sanitizer_loss_rate * Ev * Av
        return dB, dE, dA

    t = 0.0
    while t < p.cleaning_period_days - 1e-12:
        step = min(dt, p.cleaning_period_days - t)
        B, E, A = rk4_step(rhs, t, (B, E, A), step)
        t += step
    return B


def fixed_points_from_map(B0: np.ndarray, F: np.ndarray) -> list[tuple[float, float]]:
    g = F - B0
    points: list[tuple[float, float]] = []
    for i in np.where(np.signbit(g[:-1]) != np.signbit(g[1:]))[0]:
        b0, b1 = B0[i], B0[i + 1]
        g0, g1 = g[i], g[i + 1]
        root = b0 - g0 * (b1 - b0) / (g1 - g0)
        slope = (F[i + 1] - F[i]) / (b1 - b0)
        points.append((float(root), float(slope)))
    return points


def make_pulse_map_figure() -> dict[str, float]:
    p = ModelParameters()
    B0 = np.linspace(0.002, 0.98, 520)
    F_parent = daily_map_B(B0, rho=1.0, M=1.0, alpha=0.0, p=p)
    F_lys = daily_map_B(B0, rho=1.4, M=0.5, alpha=p.lysogen_loss_rate, p=p)
    parent_fp = fixed_points_from_map(B0, F_parent)
    lys_fp = fixed_points_from_map(B0, F_lys)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 5.2), constrained_layout=True)
    axes[0].plot(B0, F_parent, color=PARENT, lw=2.7, label="Parental map")
    axes[0].plot(B0, F_lys, color=LYSOGEN, lw=2.7, label="Lysogen map")
    axes[0].plot(B0, B0, color="black", lw=1.5, ls="--", label="Identity")
    axes[0].set_xlabel(r"Pre-cleaning biomass, $B_n$")
    axes[0].set_ylabel(r"Next-day biomass, $B_{n+1}$")
    axes[0].legend(loc="upper left", frameon=True, framealpha=0.95)

    axes[1].axhline(0.0, color="black", lw=1.2)
    axes[1].plot(B0, F_parent - B0, color=PARENT, lw=2.7, label="Parental")
    axes[1].plot(B0, F_lys - B0, color=LYSOGEN, lw=2.7, label="Lysogen")
    axes[1].set_xlabel(r"Pre-cleaning biomass, $B_n$")
    axes[1].set_ylabel(r"Daily biomass change, $B_{n+1}-B_n$")
    axes[1].legend(loc="upper right", frameon=True, framealpha=0.95)
    add_panel_labels(axes, ["A)", "B)"])
    for ax in axes:
        ax.grid(False)
        ax.tick_params(axis="both", which="both", colors="black")
        for spine in ax.spines.values():
            spine.set_color("black")
    apply_manuscript_typography(fig, 0.92)
    fig.savefig(FIG_DIR / "fig9_pulse_map.pdf")
    fig.savefig(FIG_DIR / "fig9_pulse_map.png")
    plt.close(fig)

    baseline_idx = int(np.argmin(np.abs(B0 - 0.82)))
    summary: dict[str, float] = {
        "pulse_map_parent_positive_fixed_count": float(len(parent_fp)),
        "pulse_map_lysogen_positive_fixed_count": float(len(lys_fp)),
        "pulse_map_parent_next_B_at_0p82": float(F_parent[baseline_idx]),
        "pulse_map_lysogen_next_B_at_0p82": float(F_lys[baseline_idx]),
        "pulse_map_parent_max_daily_multiplier": float(np.max(F_parent / B0)),
        "pulse_map_lysogen_max_daily_multiplier": float(np.max(F_lys / B0)),
    }
    if parent_fp:
        summary["pulse_map_parent_largest_fixed_B"] = max(v[0] for v in parent_fp)
        summary["pulse_map_parent_largest_fixed_slope"] = parent_fp[np.argmax([v[0] for v in parent_fp])][1]
    if lys_fp:
        summary["pulse_map_lysogen_largest_fixed_B"] = max(v[0] for v in lys_fp)
        summary["pulse_map_lysogen_largest_fixed_slope"] = lys_fp[np.argmax([v[0] for v in lys_fp])][1]
    return summary


def make_initial_maturity_figure() -> dict[str, float]:
    p = ModelParameters()
    R_M = 0.5
    R_rho = 1.4
    b0 = np.linspace(0.04, 0.95, 115)
    maturity = np.linspace(0.0, 1.6, 115)
    B0, MAT = np.meshgrid(b0, maturity)
    parent_E0 = MAT * B0
    lys_E0 = MAT * R_rho * B0
    parent_B, _, _ = integrate_monoculture_grid(
        np.ones_like(B0), np.ones_like(B0), False, p, dt=0.004, initial_B=B0, initial_E=parent_E0
    )
    lys_B, _, _ = integrate_monoculture_grid(
        np.full_like(B0, R_rho), np.full_like(B0, R_M), True, p, dt=0.004, initial_B=B0, initial_E=lys_E0
    )
    ratio = lys_B / np.maximum(parent_B, 1e-12)
    log_ratio = np.log10(np.maximum(ratio, 1e-12))

    fig, ax = plt.subplots(figsize=(7.2, 5.2), constrained_layout=True)
    levels = np.linspace(-2.0, 2.0, 41)
    cf = ax.contourf(B0, MAT, np.clip(log_ratio, -2.0, 2.0), levels=levels, cmap="BrBG")
    ax.contour(B0, MAT, ratio, levels=[1.0], colors=CONTOUR, linewidths=2.2)
    ax.scatter([0.82], [1.0], s=70, color="#c51b7d", edgecolor="white", linewidth=1.0, zorder=5, label="baseline")
    ax.set_xlabel(r"Initial viable biomass, $B_0$")
    ax.set_ylabel(r"Initial matrix maturity, $E_0/(\rho B_0)$")
    ax.grid(False)
    ax.legend(loc="upper right", frameon=True, framealpha=0.95, facecolor="white", edgecolor="#cfcfcf")
    for spine in ax.spines.values():
        spine.set_color("black")
    cbar = fig.colorbar(cf, ax=ax, pad=0.02, ticks=[-2, -1, 0, 1, 2])
    cbar.set_label(r"$\log_{10}(B_L(T)/B_S(T))$", color="black", fontsize=AXIS_LABEL_SIZE)
    cbar.ax.tick_params(colors="black", width=0.8, labelsize=TICK_LABEL_SIZE)
    cbar.outline.set_edgecolor("black")
    apply_manuscript_typography(fig, 0.82)
    fig.savefig(FIG_DIR / "fig10_initial_maturity.pdf")
    fig.savefig(FIG_DIR / "fig10_initial_maturity.png")
    plt.close(fig)
    return {
        "initial_maturity_fraction_lysogen_advantage": float(np.mean(ratio > 1.0)),
        "initial_maturity_baseline_ratio_nearest_grid": float(ratio[np.unravel_index(np.argmin((B0 - 0.82) ** 2 + (MAT - 1.0) ** 2), ratio.shape)]),
        "initial_maturity_max_ratio": float(np.max(ratio)),
        "initial_maturity_min_ratio": float(np.min(ratio)),
    }


def integrate_mixed_grid(f0: np.ndarray, R_rho: np.ndarray, p: ModelParameters, R_M: float = 0.5, dt: float = 0.004):
    total0 = 0.82
    S = total0 * (1.0 - f0)
    L = total0 * f0
    E = total0 * ((1.0 - f0) + R_rho * f0)
    A = np.zeros_like(f0, dtype=float)

    def rhs(t, y):
        Sv, Lv, Ev, Av = (np.maximum(v, 0.0) for v in y)
        total = Sv + Lv
        if t < p.cleaning_start_day:
            clean = np.zeros_like(Sv, dtype=bool)
        else:
            clean = ((t - p.cleaning_start_day) % p.cleaning_period_days) < p.cleaning_duration_days
        clean_float = np.asarray(clean, dtype=float)
        u = p.cleaning_sanitizer_input_rate * clean_float
        delta = p.cleaning_matrix_removal_rate * clean_float
        growth = np.maximum(0.0, 1.0 - total)
        kill_s = p.max_killing_rate * (Av / (Av + 1.0 + 1e-12)) / (1.0 + p.matrix_protection_strength * Ev)
        kill_l = p.max_killing_rate * (Av / (Av + R_M + 1e-12)) / (1.0 + p.matrix_protection_strength * Ev)
        dS = p.growth_rate * Sv * growth - kill_s * Sv
        dL = p.growth_rate * Lv * growth - kill_l * Lv - p.lysogen_loss_rate * Lv
        dE = p.matrix_relaxation_rate * (Sv + R_rho * Lv - Ev) - p.matrix_background_loss_rate * Ev - delta * Ev
        dA = u - p.sanitizer_washout_rate * Av - p.matrix_sanitizer_loss_rate * Ev * Av
        return dS, dL, dE, dA

    def rk4_mixed(t, y, step):
        k1 = rhs(t, y)
        y2 = tuple(yi + 0.5 * step * ki for yi, ki in zip(y, k1))
        k2 = rhs(t + 0.5 * step, y2)
        y3 = tuple(yi + 0.5 * step * ki for yi, ki in zip(y, k2))
        k3 = rhs(t + 0.5 * step, y3)
        y4 = tuple(yi + step * ki for yi, ki in zip(y, k3))
        k4 = rhs(t + step, y4)
        return tuple(np.maximum(yi + step * (ki1 + 2 * ki2 + 2 * ki3 + ki4) / 6.0, 0.0)
                     for yi, ki1, ki2, ki3, ki4 in zip(y, k1, k2, k3, k4))

    t = 0.0
    while t < p.final_time_days - 1e-12:
        step = min(dt, p.final_time_days - t)
        S, L, E, A = rk4_mixed(t, (S, L, E, A), step)
        t += step
    return S, L, E, A


def make_mixed_culture_figure() -> dict[str, float]:
    p = ModelParameters()
    f_vals = np.linspace(0.02, 0.98, 120)
    rr_vals = np.linspace(1.0, 2.8, 115)
    F0, RR = np.meshgrid(f_vals, rr_vals)
    S, L, _, _ = integrate_mixed_grid(F0, RR, p)
    final_total = S + L
    fT = L / np.maximum(final_total, 1e-12)
    logit_change = np.log10(np.maximum(fT, 1e-8) / np.maximum(1.0 - fT, 1e-8)) - np.log10(F0 / (1.0 - F0))
    parent_only, _, _ = integrate_monoculture_grid(np.ones_like(F0), np.ones_like(F0), False, p, dt=0.004)
    total_ratio = final_total / np.maximum(parent_only, 1e-12)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 5.2), constrained_layout=True)
    levels_total = np.linspace(0.5, 2.2, 35)
    cf0 = axes[0].contourf(F0, RR, np.clip(total_ratio, 0.5, 2.2), levels=levels_total, cmap="YlGnBu")
    if np.nanmin(total_ratio) <= 1.0 <= np.nanmax(total_ratio):
        axes[0].contour(F0, RR, total_ratio, levels=[1.0], colors=CONTOUR, linewidths=2.0)
    axes[0].set_xlabel(r"Initial lysogen fraction, $f_0$")
    axes[0].set_ylabel(r"Matrix ratio, $R_\rho$")
    cbar0 = fig.colorbar(cf0, ax=axes[0], pad=0.02)
    cbar0.ax.tick_params(labelsize=TICK_LABEL_SIZE)
    cbar0.set_label(r"$(S(T)+L(T))/B_S(T)$", fontsize=AXIS_LABEL_SIZE)

    levels_sel = np.linspace(-2.0, 2.0, 41)
    cf1 = axes[1].contourf(F0, RR, np.clip(logit_change, -2.0, 2.0), levels=levels_sel, cmap="BrBG")
    if np.nanmin(logit_change) <= 0.0 <= np.nanmax(logit_change):
        axes[1].contour(F0, RR, logit_change, levels=[0.0], colors=CONTOUR, linewidths=2.0)
    axes[1].set_xlabel(r"Initial lysogen fraction, $f_0$")
    axes[1].set_ylabel(r"Matrix ratio, $R_\rho$")
    cbar1 = fig.colorbar(cf1, ax=axes[1], pad=0.02, ticks=[-2, -1, 0, 1, 2])
    cbar1.ax.tick_params(labelsize=TICK_LABEL_SIZE)
    cbar1.set_label("Logit change in lysogen fraction", fontsize=AXIS_LABEL_SIZE)
    add_panel_labels(axes, ["A)", "B)"])

    for ax in axes:
        ax.grid(False)
        ax.tick_params(axis="both", which="both", colors="black")
        for spine in ax.spines.values():
            spine.set_color("black")
    apply_manuscript_typography(fig, 0.94)
    fig.savefig(FIG_DIR / "fig11_mixed_culture.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig11_mixed_culture.png", bbox_inches="tight")
    plt.close(fig)
    baseline_idx = np.unravel_index(np.argmin((F0 - 0.5) ** 2 + (RR - 1.4) ** 2), F0.shape)
    return {
        "mixed_fraction_total_above_parental_monoculture": float(np.mean(total_ratio > 1.0)),
        "mixed_fraction_lysogen_enriched": float(np.mean(logit_change > 0.0)),
        "mixed_baseline_total_ratio_nearest_grid": float(total_ratio[baseline_idx]),
        "mixed_baseline_logit_change_nearest_grid": float(logit_change[baseline_idx]),
        "mixed_baseline_final_lysogen_fraction": float(fT[baseline_idx]),
        "mixed_final_total_biomass_baseline": float(final_total[baseline_idx]),
    }

def simulate_pde_penetration(e_profile: np.ndarray, x: np.ndarray, pulse_h: float = 2.0):
    dx = x[1] - x[0]
    D0 = 0.09
    xi = 2.0
    lam = 0.30
    sink = 1.25
    bulk = 10.0
    pulse = pulse_h / 24.0
    dt = 8.0e-5
    D = D0 / (1.0 + xi * e_profile)
    D_half = 0.5 * (D[:-1] + D[1:])
    a = np.zeros_like(x)
    store_h = np.array([0.5, 1.0, 2.0])
    store_t = store_h / 24.0
    profiles: list[np.ndarray] = []
    avg_t: list[float] = []
    avg_a: list[float] = []
    t = 0.0
    store_index = 0
    while t < pulse - 1e-12:
        step = min(dt, pulse - t)
        a[-1] = bulk
        da = np.zeros_like(a)
        da[0] = D_half[0] * (a[1] - a[0]) / dx**2 - (lam + sink * e_profile[0]) * a[0]
        flux_right = D_half[1:] * (a[2:] - a[1:-1])
        flux_left = D_half[:-1] * (a[1:-1] - a[:-2])
        da[1:-1] = (flux_right - flux_left) / dx**2 - (lam + sink * e_profile[1:-1]) * a[1:-1]
        a[:-1] = np.maximum(a[:-1] + step * da[:-1], 0.0)
        a[-1] = bulk
        t += step
        avg_t.append(t * 24.0)
        avg_a.append(float(np.trapezoid(a, x)))
        while store_index < len(store_t) and t >= store_t[store_index] - 1e-12:
            profiles.append(a.copy())
            store_index += 1
    return np.array(profiles), np.array(avg_t), np.array(avg_a)


def make_pde_penetration_figure() -> dict[str, float]:
    x = np.linspace(0.0, 1.0, 120)
    parent_e = 1.0 + 0.25 * (1.0 - x)
    lysogen_e = 1.4 * parent_e
    parent_profiles, t_h, parent_avg = simulate_pde_penetration(parent_e, x)
    lys_profiles, _, lys_avg = simulate_pde_penetration(lysogen_e, x)
    colors = ["#124f70", "#2a7f62", "#b4641d"]
    labels = ["0.5 h", "1 h", "2 h"]

    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.6), constrained_layout=True)
    axes[0].plot(x, parent_e, color=PARENT, lw=2.6, label="Parental matrix")
    axes[0].plot(x, lysogen_e, color=LYSOGEN, lw=2.6, ls="--", label="Lysogen matrix")
    axes[0].set_xlabel(r"Depth from surface, $x$")
    axes[0].set_ylabel(r"Fixed matrix profile, $e(x)$")
    axes[0].legend(loc="upper right", frameon=True, framealpha=0.95)

    for i, (color, label) in enumerate(zip(colors, labels)):
        axes[1].plot(x, parent_profiles[i], color=color, lw=2.4, label=f"Parent {label}")
        axes[1].plot(x, lys_profiles[i], color=color, lw=2.4, ls="--", label=f"Lysogen {label}")
    axes[1].set_xlabel(r"Depth from surface, $x$")
    axes[1].set_ylabel(r"Sanitizer concentration, $a(x)$")
    axes[1].legend(loc="upper left", frameon=True, framealpha=0.92, fontsize=LEGEND_FONT_SIZE)

    axes[2].plot(t_h, parent_avg, color=PARENT, lw=2.7, label="Parental")
    axes[2].plot(t_h, lys_avg, color=LYSOGEN, lw=2.7, label="Lysogen")
    axes[2].set_xlabel(r"Time during cleaning (h)")
    axes[2].set_ylabel(r"Depth-averaged sanitizer, $\bar a$")
    axes[2].legend(loc="lower right", frameon=True, framealpha=0.95)
    add_panel_labels(axes, ["A)", "B)", "C)"], x=-0.10, y=1.04)
    for ax in axes:
        ax.grid(False)
        ax.tick_params(axis="both", which="both", colors="black")
        for spine in ax.spines.values():
            spine.set_color("black")
    apply_manuscript_typography(fig, 0.98)
    fig.savefig(FIG_DIR / "fig12_pde_penetration.pdf")
    fig.savefig(FIG_DIR / "fig12_pde_penetration.png")
    plt.close(fig)

    fig_a, ax_a = plt.subplots(figsize=(7.2, 5.2), constrained_layout=True)
    ax_a.plot(x, parent_e, color=PARENT, lw=2.6, label="Parental matrix")
    ax_a.plot(x, lysogen_e, color=LYSOGEN, lw=2.6, ls="--", label="Lysogen matrix")
    ax_a.set_xlabel(r"Depth from surface, $x$")
    ax_a.set_ylabel(r"Fixed matrix profile, $e(x)$")
    ax_a.legend(loc="upper right", frameon=True, framealpha=0.95)
    ax_a.grid(False)
    fig_a.savefig(FIG_DIR / "fig12a_matrix_profiles.pdf")
    fig_a.savefig(FIG_DIR / "fig12a_matrix_profiles.png")
    plt.close(fig_a)

    fig_b, ax_b = plt.subplots(figsize=(7.2, 5.2), constrained_layout=True)
    for i, (color, label) in enumerate(zip(colors, labels)):
        ax_b.plot(x, parent_profiles[i], color=color, lw=2.4, label=f"Parent {label}")
        ax_b.plot(x, lys_profiles[i], color=color, lw=2.4, ls="--", label=f"Lysogen {label}")
    ax_b.set_xlabel(r"Depth from surface, $x$")
    ax_b.set_ylabel(r"Sanitizer concentration, $a(x)$")
    ax_b.legend(loc="upper left", frameon=True, framealpha=0.92, fontsize=LEGEND_FONT_SIZE)
    ax_b.grid(False)
    fig_b.savefig(FIG_DIR / "fig12b_penetration_profiles.pdf")
    fig_b.savefig(FIG_DIR / "fig12b_penetration_profiles.png")
    plt.close(fig_b)

    fig_c, ax_c = plt.subplots(figsize=(7.2, 5.2), constrained_layout=True)
    ax_c.plot(t_h, parent_avg, color=PARENT, lw=2.7, label="Parental")
    ax_c.plot(t_h, lys_avg, color=LYSOGEN, lw=2.7, label="Lysogen")
    ax_c.set_xlabel(r"Time during cleaning (h)")
    ax_c.set_ylabel(r"Depth-averaged sanitizer, $\bar a$")
    ax_c.legend(loc="lower right", frameon=True, framealpha=0.95)
    ax_c.grid(False)
    fig_c.savefig(FIG_DIR / "fig12c_mean_exposure.pdf")
    fig_c.savefig(FIG_DIR / "fig12c_mean_exposure.png")
    plt.close(fig_c)
    return {
        "pde_parent_mean_exposure_2h": float(parent_avg[-1]),
        "pde_lysogen_mean_exposure_2h": float(lys_avg[-1]),
        "pde_lysogen_to_parent_mean_exposure_2h": float(lys_avg[-1] / max(parent_avg[-1], 1e-12)),
        "pde_parent_surface_concentration_2h": float(parent_profiles[-1, 0]),
        "pde_lysogen_surface_concentration_2h": float(lys_profiles[-1, 0]),
    }

def write_summary(
    sim_summary: dict[str, float],
    map_summary: dict[str, float],
    exposure_summary: dict[str, float],
    cleaning_summary: dict[str, float],
    sensitivity_summary: dict[str, float],
    time_summary: dict[str, float],
    boundary_summary: dict[str, float],
    pulse_summary: dict[str, float],
    maturity_summary: dict[str, float],
    mixed_summary: dict[str, float],
    pde_summary: dict[str, float],
    theta: float = 5.0,
) -> None:
    lines = []
    title = "Reproducible calculations for the prophage-biofilm model"
    lines.append(title)
    lines.append("=" * len(title))
    lines.append("")
    lines.append("Analytical threshold")
    lines.append("--------------------")
    lines.append("Matrix protection offsets the intrinsic MIC cost when")
    lines.append("    (1 + theta R_rho)/(1 + theta) > (1 + a)/(R_M + a).")
    lines.append(f"The figures use theta = {theta} unless otherwise stated.")
    lines.append("")
    for a in [1.0, 10.0]:
        lines.append(f"Exposure a = A/M_S = {a}")
        for rm in [0.5, 0.2, 0.01, 1 / 1024]:
            lines.append(f"  R_M={rm:.6g}: threshold R_rho={threshold_rrho(a, rm, theta):.4f}")
        lines.append("")
    lines.append("Exposure-threshold curve checks")
    lines.append("-------------------------------")
    for key, value in exposure_summary.items():
        lines.append(f"{key}: {value:.6g}")
    lines.append("")
    lines.append("Repeated-cleaning simulation")
    lines.append("----------------------------")
    for key, value in sim_summary.items():
        lines.append(f"{key}: {value:.6g}")
    lines.append("")
    lines.append("Survival-ratio parameter map")
    lines.append("----------------------------")
    for key, value in map_summary.items():
        lines.append(f"{key}: {value:.6g}")
    lines.append("")
    lines.append("Cleaning-control map")
    lines.append("--------------------")
    for key, value in cleaning_summary.items():
        lines.append(f"{key}: {value:.6g}")
    lines.append("")
    lines.append("Global sensitivity")
    lines.append("------------------")
    for key, value in sensitivity_summary.items():
        lines.append(f"{key}: {value:.6g}")
    lines.append("")
    lines.append("Time-to-control map")
    lines.append("-------------------")
    for key, value in time_summary.items():
        lines.append(f"{key}: {value:.6g}")
    lines.append("")
    lines.append("Boundary-shift analysis")
    lines.append("-----------------------")
    for key, value in boundary_summary.items():
        lines.append(f"{key}: {value:.6g}")
    lines.append("")
    lines.append("Pulse-map analysis")
    lines.append("------------------")
    for key, value in pulse_summary.items():
        lines.append(f"{key}: {value:.6g}")
    lines.append("")
    lines.append("Initial-maturity robustness")
    lines.append("---------------------------")
    for key, value in maturity_summary.items():
        lines.append(f"{key}: {value:.6g}")
    lines.append("")
    lines.append("Mixed-culture public-good analysis")
    lines.append("----------------------------------")
    for key, value in mixed_summary.items():
        lines.append(f"{key}: {value:.6g}")
    lines.append("")
    lines.append("PDE penetration demonstration")
    lines.append("-----------------------------")
    for key, value in pde_summary.items():
        lines.append(f"{key}: {value:.6g}")
    lines.append("")
    lines.append("Interpretation")
    lines.append("--------------")
    lines.append("The calculations show that the lysogen advantage is conditional. It can appear")
    lines.append("inside the phenotype range reported by Zhou et al. when matrix protection is")
    lines.append("strong enough, but it can be removed by longer cleaning or stronger matrix")
    lines.append("removal during cleaning.")
    (RES_DIR / "summary.txt").write_text("\n".join(lines) + "\n")


def main() -> None:
    theta = 5.0
    make_phase_figure(theta=theta)
    sim_summary = make_simulation_figure()
    map_summary = make_survival_map()
    exposure_summary = make_exposure_threshold_figure(theta=theta)
    cleaning_summary = make_cleaning_control_map()
    sensitivity_summary = make_global_sensitivity_figure()
    time_summary = make_time_to_control_figure()
    boundary_summary = make_boundary_shift_figure()
    pulse_summary = make_pulse_map_figure()
    maturity_summary = make_initial_maturity_figure()
    mixed_summary = make_mixed_culture_figure()
    pde_summary = make_pde_penetration_figure()
    write_summary(
        sim_summary,
        map_summary,
        exposure_summary,
        cleaning_summary,
        sensitivity_summary,
        time_summary,
        boundary_summary,
        pulse_summary,
        maturity_summary,
        mixed_summary,
        pde_summary,
        theta=theta,
    )
    print((RES_DIR / "summary.txt").read_text())


if __name__ == "__main__":
    main()
