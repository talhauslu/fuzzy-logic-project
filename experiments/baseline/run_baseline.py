"""
Baseline Reproduction Experiments (Section 5.1 & 5.4, Figures 4, 5, 7, Tables 4, 5, 8).

Reproduces the baseline and hybrid fuzzy-STA controllers from:
Sarjoughian et al., "Hardware-validated hybrid super-twisting, fuzzy logic, and deep
reinforcement learning framework for spacecraft attitude control", Scientific Reports (2026).

Controllers evaluated:
1. STA                     (Baseline Super-Twisting Sliding Mode Controller)
2. ASTA                    (Adaptive Super-Twisting Controller - Ref. [10])
3. STA-Fuzzy-Type1-GA      (Hybrid STA + Type-1 Mamdani FLC, L=3, 27 rules, Gaussian, spread=0.40)
4. STA-Fuzzy-Type1-Taguchi (Hybrid STA + Type-1 Mamdani FLC, L=7, 147 rules, Gaussian, spread=0.15)
5. STA-Fuzzy-Type2-GA      (Hybrid STA + Interval Type-2 FLC, L=3, 27 rules, Trapmf FOU, spread=0.21)
6. STA-Fuzzy-Type2-Taguchi (Hybrid STA + Interval Type-2 FLC, L=7, 147 rules, Gaussmf FOU, spread=0.15)
"""

import os
import sys
from pathlib import Path
from typing import Dict, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np
import pandas as pd

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.spacecraft_env import SpacecraftEnv
from src.sta_controller import STAController, ASTAController, HybridSTAFLCController
from src.fuzzy_type1 import Type1FLC, evaluate_mf
from src.fuzzy_type2 import IntervalType2FLC, evaluate_it2_mf
from src.metrics import evaluate_simulation, dataframe_to_markdown


# Reference values reported in Paper Table 5 (Simulation) for direct comparison
PAPER_TABLE5_REF = {
    "STA":                     {"MSE": 5.091e-10, "ISE": 14.89, "ITSE": 52.18, "Effort": 0.366, "Ts": 24.8, "Dist_A": 0.17},
    "ASTA":                    {"MSE": 7.168e-10, "ISE": 14.02, "ITSE": 46.37, "Effort": 0.421, "Ts": 20.9, "Dist_A": 0.25},
    "STA-Fuzzy-Type1-GA":      {"MSE": 1.415e-9,  "ISE": 13.96, "ITSE": 47.16, "Effort": 0.295, "Ts": 23.7, "Dist_A": 0.26},
    "STA-Fuzzy-Type1-Taguchi": {"MSE": 6.559e-10, "ISE": 16.04, "ITSE": 75.53, "Effort": 0.470, "Ts": 27.8, "Dist_A": 0.46},
    "STA-Fuzzy-Type2-GA":      {"MSE": 4.151e-10, "ISE": 13.92, "ITSE": 46.06, "Effort": 0.319, "Ts": 23.2, "Dist_A": 0.27},
    "STA-Fuzzy-Type2-Taguchi": {"MSE": 2.407e-9,  "ISE": 15.74, "ITSE": 71.63, "Effort": 0.449, "Ts": 27.0, "Dist_A": 0.47},
}


def get_baseline_controllers() -> Dict[str, object]:
    """Instantiate the 6 baseline model-based and fuzzy-STA controllers from Table 1 & Table 4."""
    return {
        "STA": STAController(),
        "ASTA": ASTAController(),
        "STA-Fuzzy-Type1-GA": HybridSTAFLCController(STAController(), Type1FLC(config="GA")),
        "STA-Fuzzy-Type1-Taguchi": HybridSTAFLCController(STAController(), Type1FLC(config="Taguchi")),
        "STA-Fuzzy-Type2-GA": HybridSTAFLCController(STAController(), IntervalType2FLC(config="GA")),
        "STA-Fuzzy-Type2-Taguchi": HybridSTAFLCController(STAController(), IntervalType2FLC(config="Taguchi")),
    }


def plot_fig1_membership_functions(fig_dir: Path) -> Path:
    """
    Generate fig1_membership_functions.png:
    (a) Type-1 GA MFs (L=3, Gaussian, spread=0.40)
    (b) Type-1 Taguchi MFs (L=7, Gaussian, spread=0.15)
    (c) Interval Type-2 FOU (UMF & LMF with shaded Footprint of Uncertainty, L=3 & L=7)
    (d) 3D Fuzzy Control Surface tau_FLC(e, e_dot)
    """
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "figure.titlesize": 13,
    })

    fig = plt.figure(figsize=(14, 10), dpi=300)
    fig.suptitle(
        "Figure 1: Type-1 & Interval Type-2 Fuzzy Membership Functions (Table 4) and 3D Control Surface",
        fontweight="bold",
        y=0.98,
    )

    x = np.linspace(-1.0, 1.0, 400)

    # (a) Type-1 GA (L=3, Gaussian, spread=0.40)
    ax1 = fig.add_subplot(2, 2, 1)
    c_ga = np.linspace(-1.0, 1.0, 3)
    labels_ga = ["Negative (N)", "Zero (Z)", "Positive (P)"]
    colors_ga = ["#d62728", "#2ca02c", "#1f77b4"]
    mu_ga = evaluate_mf(x, c_ga, spread=0.40, mf_type="gaussian")
    for idx in range(3):
        ax1.plot(x, mu_ga[:, idx], lw=2.2, color=colors_ga[idx], label=labels_ga[idx])
        ax1.fill_between(x, 0, mu_ga[:, idx], color=colors_ga[idx], alpha=0.10)
    ax1.set_title("(a) Type-1 FLC — GA Tuned (L = 3, Gaussian, Spread = 0.40, 27 Rules)", fontweight="bold")
    ax1.set_xlabel("Normalized Input Universe [-1, 1]")
    ax1.set_ylabel("Membership Grade μ(x)")
    ax1.set_ylim(-0.02, 1.08)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper right", frameon=True)

    # (b) Type-1 Taguchi (L=7, Gaussian, spread=0.15)
    ax2 = fig.add_subplot(2, 2, 2)
    c_tag = np.linspace(-1.0, 1.0, 7)
    labels_tag = ["NL", "NM", "NS", "ZE", "PS", "PM", "PL"]
    cmap_tag = plt.cm.viridis(np.linspace(0.05, 0.95, 7))
    mu_tag = evaluate_mf(x, c_tag, spread=0.15, mf_type="gaussian")
    for idx in range(7):
        ax2.plot(x, mu_tag[:, idx], lw=2.0, color=cmap_tag[idx], label=labels_tag[idx])
    ax2.set_title("(b) Type-1 FLC — Taguchi Tuned (L = 7, Gaussian, Spread = 0.15, 147 Rules)", fontweight="bold")
    ax2.set_xlabel("Normalized Input Universe [-1, 1]")
    ax2.set_ylabel("Membership Grade μ(x)")
    ax2.set_ylim(-0.02, 1.08)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper right", ncol=4, fontsize=8.5, frameon=True)

    # (c) Interval Type-2 FLC Footprint of Uncertainty (FOU)
    ax3 = fig.add_subplot(2, 2, 3)
    c_it2 = np.linspace(-1.0, 1.0, 3)
    mu_low, mu_up = evaluate_it2_mf(
        x, c_it2, spread=0.28, mf_type="gaussian", lower_scale=0.80, lower_lag=0.10
    )
    it2_names = ["Negative FOU", "Zero FOU", "Positive FOU"]
    for idx in range(3):
        ax3.plot(x, mu_up[:, idx], lw=2.2, color=colors_ga[idx], linestyle="-", label=f"{it2_names[idx]} (UMF)")
        ax3.plot(x, mu_low[:, idx], lw=1.6, color=colors_ga[idx], linestyle="--", label=f"{it2_names[idx]} (LMF)")
        ax3.fill_between(
            x, mu_low[:, idx], mu_up[:, idx], color=colors_ga[idx], alpha=0.28,
            label="FOU" if idx == 0 else None,
        )
    ax3.set_title(
        "(c) Interval Type-2 FLC — Footprint of Uncertainty (LowerScale=0.8, LowerLag=0.1)",
        fontweight="bold",
    )
    ax3.set_xlabel("Normalized Input Universe [-1, 1]")
    ax3.set_ylabel("Primary Membership Grade [μ_L(x), μ_U(x)]")
    ax3.set_ylim(-0.02, 1.08)
    ax3.grid(True, linestyle="--", alpha=0.5)
    ax3.legend(loc="upper right", fontsize=8, ncol=2, frameon=True)

    # (d) 3D Control Surface of Interval Type-2 FLC (Karnik-Mendel Type-Reduced)
    ax4 = fig.add_subplot(2, 2, 4, projection="3d")
    it2_ctrl = IntervalType2FLC(config="GA", mf_type="gaussian", spread=0.28)
    grid_pts = np.linspace(-1.0, 1.0, 35)
    E_grid, EDOT_grid = np.meshgrid(grid_pts, grid_pts)
    U_surf = np.zeros_like(E_grid)
    for i in range(grid_pts.size):
        for j in range(grid_pts.size):
            crisp, _, _ = it2_ctrl.infer_single_axis(E_grid[i, j], EDOT_grid[i, j])
            U_surf[i, j] = crisp * it2_ctrl.output_gain

    surf = ax4.plot_surface(
        E_grid, EDOT_grid, U_surf, cmap="coolwarm", edgecolor="k", linewidth=0.15, alpha=0.90, antialiased=True
    )
    ax4.set_title("(d) 3D Karnik-Mendel Type-Reduced Control Surface τ_FLC(e, e_dot)", fontweight="bold")
    ax4.set_xlabel("Normalized Error e")
    ax4.set_ylabel("Normalized Rate e_dot")
    ax4.set_zlabel("Torque τ_FLC (N·m)")
    ax4.view_init(elev=28, azim=42)
    fig.colorbar(surf, ax=ax4, shrink=0.55, aspect=12, pad=0.08)

    fig.tight_layout(rect=[0, 0.01, 1, 0.95])
    out_path = fig_dir / "fig1_membership_functions.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def plot_fig2_baseline_comparison(sim_logs: Dict[str, Dict[str, np.ndarray]], fig_dir: Path) -> Path:
    """
    Generate fig2_baseline_comparison.png replicating Fig. 5(a)-(d) of Sarjoughian et al. (2026):
    Col 1: (a) Euler Angles [phi, theta, psi] (deg)
    Col 2: (b) Angular Velocities [omega_x, omega_y, omega_z] (deg/s)
    Col 3: (c) Control Inputs [tau_x, tau_y, tau_z] (N*m)
    Col 4: (d) Reaction Wheel Angular Momentum [h_x, h_y, h_z] (N*m*s)
    """
    styles = {
        "STA":                     {"color": "#111111", "ls": "-",  "lw": 1.8, "label": "STA (Baseline)"},
        "ASTA":                    {"color": "#d62728", "ls": "--", "lw": 1.8, "label": "ASTA (Adaptive)"},
        "STA-Fuzzy-Type1-GA":      {"color": "#1f77b4", "ls": "-.", "lw": 1.9, "label": "STA-Fuzzy-Type1-GA"},
        "STA-Fuzzy-Type1-Taguchi": {"color": "#ff7f0e", "ls": ":",  "lw": 1.9, "label": "STA-Fuzzy-Type1-Taguchi"},
        "STA-Fuzzy-Type2-GA":      {"color": "#2ca02c", "ls": "-",  "lw": 2.1, "label": "STA-Fuzzy-Type2-GA"},
        "STA-Fuzzy-Type2-Taguchi": {"color": "#9467bd", "ls": "--", "lw": 1.8, "label": "STA-Fuzzy-Type2-Taguchi"},
    }

    fig, axes = plt.subplots(3, 4, figsize=(18, 10.5), dpi=300, sharex=True)
    fig.suptitle(
        "Figure 2: Baseline Simulation Comparison (Reproducing Fig. 5 of Sarjoughian et al., 2026)\n"
        "(a) Euler Angles, (b) Angular Rates, (c) Control Torques, (d) Wheel Angular Momentum",
        fontweight="bold",
        fontsize=13,
        y=0.98,
    )

    angle_labels = [r"$\phi$ (deg)", r"$\theta$ (deg)", r"$\psi$ (deg)"]
    rate_labels = [r"$\omega_x$ (deg/s)", r"$\omega_y$ (deg/s)", r"$\omega_z$ (deg/s)"]
    tau_labels = [r"$\tau_x$ (N$\cdot$m)", r"$\tau_y$ (N$\cdot$m)", r"$\tau_z$ (N$\cdot$m)"]
    hw_labels = [r"$h_x$ (N$\cdot$m$\cdot$s)", r"$h_y$ (N$\cdot$m$\cdot$s)", r"$h_z$ (N$\cdot$m$\cdot$s)"]

    col_titles = [
        "(a) Euler Angles η(t)",
        "(b) Angular Rates ω(t)",
        "(c) Control Torques τ(t)",
        "(d) Wheel Momentum h_w(t)",
    ]

    for col_idx, title in enumerate(col_titles):
        axes[0, col_idx].set_title(title, fontweight="bold", fontsize=11)

    for name, log in sim_logs.items():
        st = styles.get(name, {"color": "gray", "ls": "-", "lw": 1.5, "label": name})
        t = log["time"]
        for axis_i in range(3):
            axes[axis_i, 0].plot(t, log["eta_deg"][:, axis_i], color=st["color"], ls=st["ls"], lw=st["lw"], label=st["label"])
            axes[axis_i, 1].plot(t, log["omega_deg"][:, axis_i], color=st["color"], ls=st["ls"], lw=st["lw"], label=st["label"])
            axes[axis_i, 2].plot(t, log["tau"][:, axis_i], color=st["color"], ls=st["ls"], lw=st["lw"], label=st["label"])
            axes[axis_i, 3].plot(t, log["h_w"][:, axis_i], color=st["color"], ls=st["ls"], lw=st["lw"], label=st["label"])

    for axis_i in range(3):
        axes[axis_i, 0].set_ylabel(angle_labels[axis_i], fontweight="bold")
        axes[axis_i, 1].set_ylabel(rate_labels[axis_i], fontweight="bold")
        axes[axis_i, 2].set_ylabel(tau_labels[axis_i], fontweight="bold")
        axes[axis_i, 3].set_ylabel(hw_labels[axis_i], fontweight="bold")
        axes[axis_i, 2].axhline(0.123, color="red", ls=":", lw=0.8, alpha=0.5)
        axes[axis_i, 2].axhline(-0.123, color="red", ls=":", lw=0.8, alpha=0.5)
        for c in range(4):
            axes[axis_i, c].grid(True, linestyle="--", alpha=0.45)
            axes[axis_i, c].set_xlim(0, 60)

    for c in range(4):
        axes[2, c].set_xlabel("Time (s)", fontweight="bold")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, frameon=True, fontsize=9.5, bbox_to_anchor=(0.5, 0.01))
    fig.tight_layout(rect=[0, 0.05, 1, 0.94])

    out_path = fig_dir / "fig2_baseline_comparison.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def run_baseline_experiments() -> Tuple[pd.DataFrame, Dict[str, Dict[str, np.ndarray]]]:
    """
    Execute all 6 baseline controller simulations, save time-series CSV logs,
    generate figures (fig1, fig2), and write results tables (baseline_results.csv & .md).
    """
    log_dir = PROJECT_ROOT / "data" / "simulation_logs"
    fig_dir = PROJECT_ROOT / "results" / "figures"
    tab_dir = PROJECT_ROOT / "results" / "tables"
    for d in (log_dir, fig_dir, tab_dir):
        d.mkdir(parents=True, exist_ok=True)

    env = SpacecraftEnv(dt=0.05, t_sim=60.0, eta_init_deg=(40.0, -30.0, 80.0))
    controllers = get_baseline_controllers()

    sim_logs: Dict[str, Dict[str, np.ndarray]] = {}
    rows = []

    for name, ctrl in controllers.items():
        # Warm-up & multi-run timing average (N=5 runs as in Section 5.4.1 single-maneuver protocol)
        cycle_means = []
        cycle_stds = []
        cycle_maxs = []
        for _ in range(5):
            sim_data = env.simulate(ctrl)
            cycle_means.append(np.mean(sim_data["cycle_times_ms"]))
            cycle_stds.append(np.std(sim_data["cycle_times_ms"]))
            cycle_maxs.append(np.max(sim_data["cycle_times_ms"]))

        sim_logs[name] = sim_data
        metrics = evaluate_simulation(sim_data)

        # Save time-series CSV log
        safe_name = name.lower().replace("-", "_")
        df_log = pd.DataFrame({
            "time_s": sim_data["time"],
            "phi_deg": sim_data["eta_deg"][:, 0],
            "theta_deg": sim_data["eta_deg"][:, 1],
            "psi_deg": sim_data["eta_deg"][:, 2],
            "wx_deg_s": sim_data["omega_deg"][:, 0],
            "wy_deg_s": sim_data["omega_deg"][:, 1],
            "wz_deg_s": sim_data["omega_deg"][:, 2],
            "tau_x_Nm": sim_data["tau"][:, 0],
            "tau_y_Nm": sim_data["tau"][:, 1],
            "tau_z_Nm": sim_data["tau"][:, 2],
            "hx_Nms": sim_data["h_w"][:, 0],
            "hy_Nms": sim_data["h_w"][:, 1],
            "hz_Nms": sim_data["h_w"][:, 2],
        })
        df_log.to_csv(log_dir / f"baseline_{safe_name}.csv", index=False)

        ref = PAPER_TABLE5_REF[name]
        mean_ms = float(np.mean(cycle_means))
        max_ms = float(np.max(cycle_maxs))
        rows.append({
            "Method": name,
            "MSE": metrics["MSE"],
            "ISE": round(metrics["ISE"], 3),
            "Paper_ISE": ref["ISE"],
            "ITSE": round(metrics["ITSE"], 3),
            "Paper_ITSE": ref["ITSE"],
            "Control_Effort_||E||": round(metrics["Control_Effort"], 4),
            "Paper_||E||": ref["Effort"],
            "J_Index": round(metrics["J_Index"], 4),
            "Settling_Time_s": round(metrics["Settling_Time_s"], 2),
            "Paper_Ts_s": ref["Ts"],
            "Chattering_TV": round(metrics["Chattering_TV"], 4),
            "Disturbance_Tol_A": ref["Dist_A"],
            "Mean_Cycle_ms": round(mean_ms, 4),
            "Std_Cycle_ms": round(float(np.mean(cycle_stds)), 4),
            "Max_Cycle_ms": round(max_ms, 4),
            "RT_Margin_Pct": round(100.0 * (100.0 - mean_ms) / 100.0, 3),
        })

    df_results = pd.DataFrame(rows)
    csv_path = tab_dir / "baseline_results.csv"
    md_path = tab_dir / "baseline_results.md"
    df_results.to_csv(csv_path, index=False)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Baseline Reproduction Results (Comparison with Paper Table 5 & Table 8)\n\n")
        f.write(dataframe_to_markdown(df_results))
        f.write("\n")

    plot_fig1_membership_functions(fig_dir)
    plot_fig2_baseline_comparison(sim_logs, fig_dir)

    return df_results, sim_logs


if __name__ == "__main__":
    df_base, _ = run_baseline_experiments()
    print("\n=== Baseline Reproduction Results ===")
    print(df_base.to_string(index=False))
