"""
Proposed Extension, Sensor Noise Robustness, and Ablation Study Experiments.

Implements:
1. Proposed Adaptive Gain-Scheduled Hybrid STA-IT2FLC (Proposed-AGS-IT2FLC) vs. Baselines
   -> Saves `results/tables/proposed_comparison.csv` & `.md`
   -> Generates `results/figures/fig3_proposed_vs_baseline.png`
2. Sensor Noise & External Disturbance Robustness Analysis (Sim-to-Real HIL Emulation)
   -> Evaluates controllers under 0.2 deg attitude sensor noise, 0.35 deg/s rate noise,
      and external pulse/sinusoidal torque disturbance
   -> Generates `results/figures/fig4_noise_robustness.png`
3. Membership Function & Defuzzification Ablation Study (Table 2 & Section 3.2.1)
   -> Evaluates Gaussian, Triangular, Trapezoidal MFs & Centroid, Bisector, MOM, SOM
   -> Saves `results/tables/ablation_and_noise_results.csv` & `.md`
   -> Generates `results/figures/fig5_ablation_study.png`
4. Per-Control-Cycle Computational Load Assessment (Reproducing Fig. 7 & Table 8)
   -> Generates `results/figures/fig6_computational_timing.png`
"""

import sys
from pathlib import Path
from typing import Dict, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.spacecraft_env import SpacecraftEnv
from src.sta_controller import STAController, ASTAController, HybridSTAFLCController
from src.fuzzy_type1 import Type1FLC
from src.fuzzy_type2 import IntervalType2FLC
from src.proposed_controller import ProposedAGSIT2FLC
from src.metrics import evaluate_simulation, dataframe_to_markdown
from experiments.baseline.run_baseline import get_baseline_controllers


def plot_fig3_proposed_vs_baseline(
    sim_logs: Dict[str, Dict[str, np.ndarray]],
    fig_dir: Path,
) -> Path:
    """
    Generate fig3_proposed_vs_baseline.png:
    (a) Roll/Pitch/Yaw Attitude Error Norm ||e(t)|| (deg)
    (b) Log-Scale Tracking Error Norm ||e(t)|| (deg) showing steady-state precision
    (c) 3-Axis Euler Angles Convergence Comparison (Roll, Pitch, Yaw)
    (d) Control Torque Effort ||tau(t)|| (N*m) & Cumulative Energy Consumption
    """
    selected = [
        "STA",
        "ASTA",
        "STA-Fuzzy-Type1-GA",
        "STA-Fuzzy-Type2-GA",
        "STA-Fuzzy-Type2-Taguchi",
        "Proposed-AGS-IT2FLC",
    ]
    styles = {
        "STA":                     {"color": "#333333", "ls": "--", "lw": 1.6, "label": "STA (Baseline)"},
        "ASTA":                    {"color": "#e377c2", "ls": ":",  "lw": 1.7, "label": "ASTA (Adaptive)"},
        "STA-Fuzzy-Type1-GA":      {"color": "#1f77b4", "ls": "-.", "lw": 1.8, "label": "STA-Fuzzy-Type1-GA"},
        "STA-Fuzzy-Type2-GA":      {"color": "#2ca02c", "ls": "--", "lw": 1.9, "label": "STA-Fuzzy-Type2-GA"},
        "STA-Fuzzy-Type2-Taguchi": {"color": "#9467bd", "ls": ":",  "lw": 1.8, "label": "STA-Fuzzy-Type2-Taguchi"},
        "Proposed-AGS-IT2FLC":     {"color": "#d62728", "ls": "-",  "lw": 2.6, "label": "Proposed-AGS-IT2FLC (Ours)"},
    }

    fig, axes = plt.subplots(2, 2, figsize=(15, 10), dpi=300)
    fig.suptitle(
        "Figure 3: Proposed Adaptive Gain-Scheduled Hybrid STA-IT2FLC vs. Baseline Controllers",
        fontweight="bold",
        fontsize=13,
        y=0.98,
    )

    ax_err, ax_log = axes[0, 0], axes[0, 1]
    ax_yaw, ax_energy = axes[1, 0], axes[1, 1]

    for name in selected:
        if name not in sim_logs:
            continue
        log = sim_logs[name]
        st = styles[name]
        t = log["time"]
        err_norm_deg = np.linalg.norm(log["eta_deg"], axis=1)
        dt = float(t[1] - t[0])
        cum_energy = np.cumsum(np.sum(log["tau"] ** 2, axis=1)) * dt

        ax_err.plot(t, err_norm_deg, color=st["color"], ls=st["ls"], lw=st["lw"], label=st["label"])
        ax_log.semilogy(t, np.maximum(err_norm_deg, 1e-5), color=st["color"], ls=st["ls"], lw=st["lw"], label=st["label"])
        ax_yaw.plot(t, log["eta_deg"][:, 2], color=st["color"], ls=st["ls"], lw=st["lw"], label=st["label"])
        ax_energy.plot(t, cum_energy, color=st["color"], ls=st["ls"], lw=st["lw"], label=st["label"])

    ax_err.axhline(2.0, color="black", ls=":", lw=1.0, alpha=0.7, label="±2° Settling Band")
    ax_err.set_title("(a) 3-Axis Attitude Error Norm ||e(t)|| (deg)", fontweight="bold")
    ax_err.set_xlabel("Time (s)")
    ax_err.set_ylabel("||e(t)|| (deg)")
    ax_err.set_xlim(0, 45)
    ax_err.grid(True, linestyle="--", alpha=0.5)
    ax_err.legend(loc="upper right", fontsize=8.5, frameon=True)

    ax_log.axhline(2.0, color="black", ls=":", lw=1.0, alpha=0.7)
    ax_log.set_title("(b) Log-Scale Attitude Error Norm ||e(t)|| (deg)", fontweight="bold")
    ax_log.set_xlabel("Time (s)")
    ax_log.set_ylabel("log10 ||e(t)|| (deg)")
    ax_log.set_xlim(0, 60)
    ax_log.grid(True, which="both", linestyle="--", alpha=0.45)
    ax_log.legend(loc="upper right", fontsize=8.5, frameon=True)

    ax_yaw.axhline(0.0, color="black", ls=":", lw=0.9, alpha=0.6)
    ax_yaw.set_title("(c) Largest-Angle Maneuver: Yaw ψ(t) from 80° to 0°", fontweight="bold")
    ax_yaw.set_xlabel("Time (s)")
    ax_yaw.set_ylabel("Yaw Angle ψ (deg)")
    ax_yaw.set_xlim(0, 45)
    ax_yaw.grid(True, linestyle="--", alpha=0.5)
    ax_yaw.legend(loc="upper right", fontsize=8.5, frameon=True)

    ax_energy.set_title(r"(d) Cumulative Actuator Control Energy $\int_0^t \|\tau(\tau)\|^2 d\tau$", fontweight="bold")
    ax_energy.set_xlabel("Time (s)")
    ax_energy.set_ylabel(r"Cumulative Energy ($\mathrm{N^2 \cdot m^2 \cdot s}$)")
    ax_energy.set_xlim(0, 60)
    ax_energy.grid(True, linestyle="--", alpha=0.5)
    ax_energy.legend(loc="lower right", fontsize=8.5, frameon=True)

    fig.tight_layout(rect=[0, 0.01, 1, 0.95])
    out_path = fig_dir / "fig3_proposed_vs_baseline.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def run_noise_and_disturbance_experiment(fig_dir: Path) -> pd.DataFrame:
    """
    Emulate the HIL Sim-to-Real Environment (Section 5.2, Table 7):
    - Attitude sensor noise: 0.20 deg std (matching MicroStrain 3DM-GX3-25 AHRS spec on p. 37)
    - Angular velocity noise: 0.35 deg/s std
    - External torque disturbance pulse + sinusoidal perturbation during t in [25, 40] s
    """
    noisy_env = SpacecraftEnv(
        dt=0.05,
        t_sim=60.0,
        eta_init_deg=(40.0, -30.0, 80.0),
        sensor_noise_std_deg=0.20,
        sensor_noise_rate_std_deg=0.35,
        seed=42,
    )

    def ext_disturbance(t: float) -> np.ndarray:
        # Commanded external disturbance Eq. (55) + external orbital perturbation pulse in [28, 36] s
        base_d = SpacecraftEnv.command_disturbance_torque(t, amplitude_A=0.012)
        if 28.0 <= t <= 36.0:
            pulse = np.array([0.032, -0.028, 0.035], dtype=float) * np.sin(0.8 * (t - 28.0))
            return base_d + pulse
        return base_d

    test_controllers = {
        "STA": STAController(),
        "ASTA": ASTAController(),
        "STA-Fuzzy-Type1-GA": HybridSTAFLCController(STAController(), Type1FLC("GA")),
        "STA-Fuzzy-Type2-GA": HybridSTAFLCController(STAController(), IntervalType2FLC("GA")),
        "STA-Fuzzy-Type2-Taguchi": HybridSTAFLCController(STAController(), IntervalType2FLC("Taguchi")),
        "Proposed-AGS-IT2FLC": ProposedAGSIT2FLC(),
    }

    noisy_logs = {}
    rows = []
    for name, ctrl in test_controllers.items():
        sim_data = noisy_env.simulate(ctrl, disturbance_fn=ext_disturbance)
        noisy_logs[name] = sim_data
        m = evaluate_simulation(sim_data, settling_threshold_deg=2.5)
        rows.append({
            "Experiment": "Sensor_Noise_&_Disturbance",
            "Configuration": name,
            "ISE": round(m["ISE"], 3),
            "ITSE": round(m["ITSE"], 3),
            "Control_Effort_||E||": round(m["Control_Effort"], 4),
            "J_Index": round(m["J_Index"], 4),
            "Settling_Time_s": round(m["Settling_Time_s"], 2),
            "Chattering_TV": round(m["Chattering_TV"], 3),
            "Steady_State_MSE": m["MSE_Full"],
        })

    # Plot fig4_noise_robustness.png
    styles = {
        "STA":                     {"color": "#333333", "ls": "--", "lw": 1.5, "label": "STA"},
        "ASTA":                    {"color": "#ff7f0e", "ls": ":",  "lw": 1.6, "label": "ASTA (Noise-Amplified)"},
        "STA-Fuzzy-Type1-GA":      {"color": "#1f77b4", "ls": "-.", "lw": 1.6, "label": "STA-Fuzzy-Type1-GA"},
        "STA-Fuzzy-Type2-GA":      {"color": "#2ca02c", "ls": "--", "lw": 1.8, "label": "STA-Fuzzy-Type2-GA"},
        "STA-Fuzzy-Type2-Taguchi": {"color": "#9467bd", "ls": ":",  "lw": 1.7, "label": "STA-Fuzzy-Type2-Taguchi"},
        "Proposed-AGS-IT2FLC":     {"color": "#d62728", "ls": "-",  "lw": 2.4, "label": "Proposed-AGS-IT2FLC (Ours)"},
    }

    fig, axes = plt.subplots(2, 2, figsize=(15, 9.5), dpi=300)
    fig.suptitle(
        "Figure 4: Sensor Noise (σ_η=0.2°, σ_ω=0.35°/s) & External Torque Disturbance Robustness Analysis",
        fontweight="bold",
        fontsize=13,
        y=0.98,
    )

    ax1, ax2 = axes[0, 0], axes[0, 1]
    ax3, ax4 = axes[1, 0], axes[1, 1]

    for name, log in noisy_logs.items():
        st = styles[name]
        t = log["time"]
        err_norm = np.linalg.norm(log["eta_deg"], axis=1)
        ax1.plot(t, err_norm, color=st["color"], ls=st["ls"], lw=st["lw"], label=st["label"])
        ax2.plot(t, err_norm, color=st["color"], ls=st["ls"], lw=st["lw"], label=st["label"])
        if name in ("ASTA", "STA-Fuzzy-Type1-GA", "STA-Fuzzy-Type2-GA", "Proposed-AGS-IT2FLC"):
            ax3.plot(t, log["tau"][:, 0], color=st["color"], ls=st["ls"], lw=st["lw"], alpha=0.85, label=st["label"])

    ax1.axvspan(28.0, 36.0, color="gold", alpha=0.22, label="External Disturbance Window")
    ax1.set_title("(a) Full-Horizon Attitude Error Norm ||e(t)|| Under Noise & Disturbance", fontweight="bold")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("||e(t)|| (deg)")
    ax1.set_xlim(0, 60)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper right", fontsize=8.5, frameon=True)

    ax2.axvspan(28.0, 36.0, color="gold", alpha=0.25, label="Disturbance Window [28s, 36s]")
    ax2.set_title("(b) Zoomed Steady-State & Disturbance Rejection Response (t ∈ [22s, 55s])", fontweight="bold")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("||e(t)|| (deg)")
    ax2.set_xlim(22, 55)
    ax2.set_ylim(0, 6.0)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper right", fontsize=8.5, frameon=True)

    ax3.set_title("(c) Roll Axis Control Torque τ_x(t) Chattering Under Sensor Noise", fontweight="bold")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Torque τ_x (N·m)")
    ax3.set_xlim(20, 50)
    ax3.grid(True, linestyle="--", alpha=0.5)
    ax3.legend(loc="upper right", fontsize=8.5, frameon=True)

    # Bar chart of Chattering TV & J_Index under noise
    df_noise = pd.DataFrame(rows)
    x_pos = np.arange(len(df_noise))
    width = 0.36
    ax4_twin = ax4.twinx()
    b1 = ax4.bar(x_pos - width / 2, df_noise["Chattering_TV"], width, color="#1f77b4", alpha=0.85, label="Chattering TV (N·m)")
    b2 = ax4_twin.bar(x_pos + width / 2, df_noise["J_Index"], width, color="#d62728", alpha=0.85, label="Composite Index J")
    ax4.set_title("(d) Chattering Total Variation (TV) and Composite Index J Under Noise", fontweight="bold")
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels([c.replace("STA-Fuzzy-", "F-").replace("Proposed-", "") for c in df_noise["Configuration"]], rotation=18, fontsize=8.5)
    ax4.set_ylabel("Chattering Total Variation (TV)", color="#1f77b4", fontweight="bold")
    ax4_twin.set_ylabel("Multi-Objective Index J", color="#d62728", fontweight="bold")
    ax4.grid(True, axis="y", linestyle="--", alpha=0.4)

    lines_1, labels_1 = ax4.get_legend_handles_labels()
    lines_2, labels_2 = ax4_twin.get_legend_handles_labels()
    ax4.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left", fontsize=8.5, frameon=True)

    fig.tight_layout(rect=[0, 0.01, 1, 0.95])
    fig.savefig(fig_dir / "fig4_noise_robustness.png", dpi=300)
    plt.close(fig)

    return df_noise


def run_ablation_study(fig_dir: Path) -> pd.DataFrame:
    """
    Execute Membership Function Prototype & Defuzzification Method Ablation Study (Table 2 & Section 3.2.1):
    - Part A: MF Prototype Ablation ('gaussian', 'triangular', 'trapezoidal') on Type-1 & Type-2 FLC
    - Part B: Defuzzification Method Ablation ('centroid', 'bisector', 'mom', 'som') on Type-1 FLC
    """
    env = SpacecraftEnv(dt=0.05, t_sim=60.0)
    rows = []

    # Part A: Membership Function Prototype Comparison (Table 2 A3 parameter)
    for mf_proto in ("gaussian", "triangular", "trapezoidal"):
        for flc_kind, cls in (("Type-1 (GA, L=3)", Type1FLC), ("Type-2 (GA, L=3)", IntervalType2FLC)):
            ctrl = HybridSTAFLCController(
                STAController(),
                cls(config="GA", mf_type=mf_proto),
            )
            sim_data = env.simulate(ctrl)
            m = evaluate_simulation(sim_data)
            rows.append({
                "Experiment": "MF_Prototype_Ablation",
                "Configuration": f"{flc_kind} - {mf_proto.capitalize()}",
                "ISE": round(m["ISE"], 3),
                "ITSE": round(m["ITSE"], 3),
                "Control_Effort_||E||": round(m["Control_Effort"], 4),
                "J_Index": round(m["J_Index"], 4),
                "Settling_Time_s": round(m["Settling_Time_s"], 2),
                "Chattering_TV": round(m["Chattering_TV"], 3),
                "Steady_State_MSE": m["MSE"],
            })

    # Part B: Defuzzification Method Ablation (Section 3.2.1 Eq. 15)
    for defuzz in ("centroid", "bisector", "mom", "som"):
        ctrl = HybridSTAFLCController(
            STAController(),
            Type1FLC(config="GA", mf_type="gaussian", defuzz_method=defuzz),
        )
        sim_data = env.simulate(ctrl)
        m = evaluate_simulation(sim_data)
        rows.append({
            "Experiment": "Defuzzification_Ablation",
            "Configuration": f"Type-1 GA - {defuzz.upper()}",
            "ISE": round(m["ISE"], 3),
            "ITSE": round(m["ITSE"], 3),
            "Control_Effort_||E||": round(m["Control_Effort"], 4),
            "J_Index": round(m["J_Index"], 4),
            "Settling_Time_s": round(m["Settling_Time_s"], 2),
            "Chattering_TV": round(m["Chattering_TV"], 3),
            "Steady_State_MSE": m["MSE"],
        })

    df_ablation = pd.DataFrame(rows)

    # Plot fig5_ablation_study.png
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.2), dpi=300)
    fig.suptitle(
        "Figure 5: Ablation Study — (a) Membership Function Prototypes (Table 2) & (b) Defuzzification Methods",
        fontweight="bold",
        fontsize=13,
        y=0.98,
    )

    df_mf = df_ablation[df_ablation["Experiment"] == "MF_Prototype_Ablation"]
    df_def = df_ablation[df_ablation["Experiment"] == "Defuzzification_Ablation"]

    # Subplot (a): MF Prototypes
    ax1 = axes[0]
    x1 = np.arange(len(df_mf))
    w = 0.26
    ax1.bar(x1 - w, df_mf["ITSE"] / 100.0, w, label="ITSE / 100", color="#1f77b4")
    ax1.bar(x1, df_mf["Control_Effort_||E||"], w, label="Control Effort ||E||", color="#2ca02c")
    ax1.bar(x1 + w, df_mf["J_Index"], w, label="Objective Index J (Eq. 19)", color="#d62728")
    ax1.set_title("(a) MF Prototype Ablation (Gaussian vs. Triangular vs. Trapezoidal)", fontweight="bold")
    ax1.set_xticks(x1)
    ax1.set_xticklabels([c.replace(" (GA, L=3)", "") for c in df_mf["Configuration"]], rotation=20, fontsize=8.5)
    ax1.set_ylabel("Metric Value")
    ax1.grid(True, axis="y", linestyle="--", alpha=0.5)
    ax1.legend(loc="upper left", fontsize=8.5, frameon=True)

    # Subplot (b): Defuzzification Methods
    ax2 = axes[1]
    x2 = np.arange(len(df_def))
    ax2.bar(x2 - w, df_def["ITSE"] / 100.0, w, label="ITSE / 100", color="#9467bd")
    ax2.bar(x2, df_def["Control_Effort_||E||"], w, label="Control Effort ||E||", color="#ff7f0e")
    ax2.bar(x2 + w, df_def["J_Index"], w, label="Objective Index J (Eq. 19)", color="#d62728")
    for idx, val in enumerate(df_def["Chattering_TV"]):
        ax2.text(x2[idx], max(df_def["ITSE"].iloc[idx] / 100.0, df_def["J_Index"].iloc[idx]) + 0.015,
                 f"TV={val:.2f}", ha="center", fontsize=8, fontweight="bold")
    ax2.set_title("(b) Defuzzification Method Ablation (Centroid vs. Bisector vs. MOM vs. SOM)", fontweight="bold")
    ax2.set_xticks(x2)
    ax2.set_xticklabels([c.replace("Type-1 GA - ", "") for c in df_def["Configuration"]], fontsize=9.5)
    ax2.set_ylabel("Metric Value")
    ax2.set_ylim(0, max(df_def["ITSE"] / 100.0) * 1.18)
    ax2.grid(True, axis="y", linestyle="--", alpha=0.5)
    ax2.legend(loc="upper left", fontsize=8.5, frameon=True)

    fig.tight_layout(rect=[0, 0.02, 1, 0.93])
    fig.savefig(fig_dir / "fig5_ablation_study.png", dpi=300)
    plt.close(fig)

    return df_ablation


def plot_fig6_computational_timing(df_comp: pd.DataFrame, fig_dir: Path) -> Path:
    """
    Generate fig6_computational_timing.png reproducing Fig. 7 of Sarjoughian et al. (2026):
    Bar chart of Mean per-control-cycle computation time (ms) with +/- 1 SD error bars
    and diamond markers for Maximum per-cycle time (ms).
    """
    fig, ax = plt.subplots(figsize=(12, 6.2), dpi=300)
    x = np.arange(len(df_comp))
    means = df_comp["Mean_Cycle_ms"].to_numpy()
    stds = df_comp["Std_Cycle_ms"].to_numpy()
    maxs = df_comp["Max_Cycle_ms"].to_numpy()

    bars = ax.bar(
        x, means, width=0.55, color="#0072bd", edgecolor="black", linewidth=0.8,
        yerr=stds, capsize=4, error_kw={"elinewidth": 1.3, "ecolor": "black"},
        label="Mean ± SD (ms)",
    )
    bars[-1].set_facecolor("#d62728")  # Highlight Proposed controller

    ax.scatter(
        x, maxs, marker="D", s=48, facecolors="none", edgecolors="black", linewidths=1.5,
        zorder=5, label="Maximum Cycle Time (ms)",
    )

    for idx, m_val in enumerate(means):
        ax.text(
            x[idx], maxs[idx] + max(maxs) * 0.03,
            f"{m_val:.3f} ms\n(Margin: {df_comp['RT_Margin_Pct'].iloc[idx]:.2f}%)",
            ha="center", va="bottom", fontsize=8, fontweight="bold",
        )

    ax.set_title(
        "Figure 6: Per-Control-Cycle Computational Load Comparison (Single-Maneuver Protocol, N = 5)\n"
        "Reproducing Fig. 7 & Table 8 of Sarjoughian et al. (2026) — 100 ms Real-Time Control Period",
        fontweight="bold",
        fontsize=11.5,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(df_comp["Method"], rotation=22, ha="right", fontsize=9.5, fontweight="bold")
    ax.set_ylabel("Controller Computation Time per Control Cycle (ms)", fontweight="bold")
    ax.set_ylim(0, max(maxs) * 1.25)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="upper left", frameon=True, fontsize=9.5)

    fig.tight_layout()
    out_path = fig_dir / "fig6_computational_timing.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def run_proposed_experiments(
    baseline_df: pd.DataFrame = None,
    baseline_logs: Dict[str, Dict[str, np.ndarray]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Execute all Proposed Extension, Noise Robustness, Ablation, and Timing experiments.
    """
    log_dir = PROJECT_ROOT / "data" / "simulation_logs"
    fig_dir = PROJECT_ROOT / "results" / "figures"
    tab_dir = PROJECT_ROOT / "results" / "tables"
    for d in (log_dir, fig_dir, tab_dir):
        d.mkdir(parents=True, exist_ok=True)

    if baseline_df is None or baseline_logs is None:
        from experiments.baseline.run_baseline import run_baseline_experiments
        baseline_df, baseline_logs = run_baseline_experiments()

    env = SpacecraftEnv(dt=0.05, t_sim=60.0, eta_init_deg=(40.0, -30.0, 80.0))
    prop_ctrl = ProposedAGSIT2FLC()

    cycle_means, cycle_stds, cycle_maxs = [], [], []
    for _ in range(5):
        prop_sim = env.simulate(prop_ctrl)
        cycle_means.append(np.mean(prop_sim["cycle_times_ms"]))
        cycle_stds.append(np.std(prop_sim["cycle_times_ms"]))
        cycle_maxs.append(np.max(prop_sim["cycle_times_ms"]))

    prop_metrics = evaluate_simulation(prop_sim)

    # Save Proposed simulation log CSV
    df_prop_log = pd.DataFrame({
        "time_s": prop_sim["time"],
        "phi_deg": prop_sim["eta_deg"][:, 0],
        "theta_deg": prop_sim["eta_deg"][:, 1],
        "psi_deg": prop_sim["eta_deg"][:, 2],
        "wx_deg_s": prop_sim["omega_deg"][:, 0],
        "wy_deg_s": prop_sim["omega_deg"][:, 1],
        "wz_deg_s": prop_sim["omega_deg"][:, 2],
        "tau_x_Nm": prop_sim["tau"][:, 0],
        "tau_y_Nm": prop_sim["tau"][:, 1],
        "tau_z_Nm": prop_sim["tau"][:, 2],
        "hx_Nms": prop_sim["h_w"][:, 0],
        "hy_Nms": prop_sim["h_w"][:, 1],
        "hz_Nms": prop_sim["h_w"][:, 2],
    })
    df_prop_log.to_csv(log_dir / "proposed_ags_it2flc.csv", index=False)

    all_logs = dict(baseline_logs)
    all_logs["Proposed-AGS-IT2FLC"] = prop_sim

    # Build proposed_comparison DataFrame
    comp_rows = []
    sta_row = baseline_df[baseline_df["Method"] == "STA"].iloc[0]
    for _, r in baseline_df.iterrows():
        comp_rows.append({
            "Method": r["Method"],
            "MSE": r["MSE"],
            "ISE": r["ISE"],
            "ITSE": r["ITSE"],
            "Control_Effort_||E||": r["Control_Effort_||E||"],
            "J_Index": r["J_Index"],
            "Settling_Time_s": r["Settling_Time_s"],
            "Chattering_TV": r["Chattering_TV"],
            "Mean_Cycle_ms": r["Mean_Cycle_ms"],
            "Std_Cycle_ms": r["Std_Cycle_ms"],
            "Max_Cycle_ms": r["Max_Cycle_ms"],
            "RT_Margin_Pct": r["RT_Margin_Pct"],
            "J_Improvement_vs_STA_%": round(100.0 * (sta_row["J_Index"] - r["J_Index"]) / sta_row["J_Index"], 2),
        })

    mean_ms = float(np.mean(cycle_means))
    max_ms = float(np.max(cycle_maxs))
    comp_rows.append({
        "Method": "Proposed-AGS-IT2FLC",
        "MSE": prop_metrics["MSE"],
        "ISE": round(prop_metrics["ISE"], 3),
        "ITSE": round(prop_metrics["ITSE"], 3),
        "Control_Effort_||E||": round(prop_metrics["Control_Effort"], 4),
        "J_Index": round(prop_metrics["J_Index"], 4),
        "Settling_Time_s": round(prop_metrics["Settling_Time_s"], 2),
        "Chattering_TV": round(prop_metrics["Chattering_TV"], 4),
        "Mean_Cycle_ms": round(mean_ms, 4),
        "Std_Cycle_ms": round(float(np.mean(cycle_stds)), 4),
        "Max_Cycle_ms": round(max_ms, 4),
        "RT_Margin_Pct": round(100.0 * (100.0 - mean_ms) / 100.0, 3),
        "J_Improvement_vs_STA_%": round(100.0 * (sta_row["J_Index"] - prop_metrics["J_Index"]) / sta_row["J_Index"], 2),
    })

    df_proposed_comp = pd.DataFrame(comp_rows)
    df_proposed_comp.to_csv(tab_dir / "proposed_comparison.csv", index=False)
    with open(tab_dir / "proposed_comparison.md", "w", encoding="utf-8") as f:
        f.write("# Proposed Adaptive Gain-Scheduled Hybrid STA-IT2FLC vs. Baseline Comparison\n\n")
        f.write(dataframe_to_markdown(df_proposed_comp))
        f.write("\n")

    # Generate Figure 3 & Figure 6
    plot_fig3_proposed_vs_baseline(all_logs, fig_dir)
    plot_fig6_computational_timing(df_proposed_comp, fig_dir)

    # Run Sensor Noise & Disturbance Robustness + Ablation Study (Figure 4 & Figure 5)
    df_noise = run_noise_and_disturbance_experiment(fig_dir)
    df_ablation = run_ablation_study(fig_dir)

    df_combined_ext = pd.concat([df_noise, df_ablation], ignore_index=True)
    df_combined_ext.to_csv(tab_dir / "ablation_and_noise_results.csv", index=False)
    with open(tab_dir / "ablation_and_noise_results.md", "w", encoding="utf-8") as f:
        f.write("# Sensor Noise Robustness & MF/Defuzzification Ablation Study Results\n\n")
        f.write("## 1. Sensor Noise (σ_η = 0.2°, σ_ω = 0.35°/s) & External Torque Disturbance Robustness\n\n")
        f.write(dataframe_to_markdown(df_noise))
        f.write("\n\n## 2. Membership Function Prototype & Defuzzification Method Ablation Study\n\n")
        f.write(dataframe_to_markdown(df_ablation))
        f.write("\n")

    return df_proposed_comp, df_combined_ext


if __name__ == "__main__":
    df_p, df_ext = run_proposed_experiments()
    print("\n=== Proposed vs. Baseline Comparison ===")
    print(df_p.to_string(index=False))
    print("\n=== Ablation & Sensor Noise Robustness Results ===")
    print(df_ext.to_string(index=False))
