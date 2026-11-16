"""
Interactive Live Presentation & Defense Demo Script (`interactive_demo.py`).

Allows the presenter / instructor during the oral defense (Section 14 of Project Guidelines)
to test custom initial Euler angles, sensor noise levels, external disturbance torques,
membership function prototypes, and defuzzification methods on the fly via CLI arguments
or an interactive terminal menu, and generates an instant comparison plot.

Usage Examples:
    # 1. Interactive Menu Mode:
    python interactive_demo.py

    # 2. Direct CLI Flag Mode (e.g., custom angles [65, -40, 110], sensor noise 0.25 deg):
    python interactive_demo.py --angles 65 -40 110 --noise 0.25 --disturbance 0.02 --compare-all
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.spacecraft_env import SpacecraftEnv
from src.sta_controller import STAController, ASTAController, HybridSTAFLCController
from src.fuzzy_type1 import Type1FLC
from src.fuzzy_type2 import IntervalType2FLC
from src.proposed_controller import ProposedAGSIT2FLC
from src.metrics import evaluate_simulation, dataframe_to_markdown


def build_controller(
    name: str,
    mf_type: str = None,
    defuzz_method: str = "centroid",
):
    """Instantiate a controller by name with optional custom MF prototype and defuzzification."""
    key = name.upper().strip()
    if key in ("1", "STA"):
        return "STA", STAController()
    elif key in ("2", "ASTA"):
        return "ASTA", ASTAController()
    elif key in ("3", "STA-T1-GA", "STA-FUZZY-TYPE1-GA"):
        return "STA-Fuzzy-Type1-GA", HybridSTAFLCController(
            STAController(),
            Type1FLC(config="GA", mf_type=mf_type or "gaussian", defuzz_method=defuzz_method),
        )
    elif key in ("4", "STA-T1-TAGUCHI", "STA-FUZZY-TYPE1-TAGUCHI"):
        return "STA-Fuzzy-Type1-Taguchi", HybridSTAFLCController(
            STAController(),
            Type1FLC(config="Taguchi", mf_type=mf_type or "gaussian", defuzz_method=defuzz_method),
        )
    elif key in ("5", "STA-T2-GA", "STA-FUZZY-TYPE2-GA"):
        return "STA-Fuzzy-Type2-GA", HybridSTAFLCController(
            STAController(),
            IntervalType2FLC(config="GA", mf_type=mf_type or "trapezoidal"),
        )
    elif key in ("6", "STA-T2-TAGUCHI", "STA-FUZZY-TYPE2-TAGUCHI"):
        return "STA-Fuzzy-Type2-Taguchi", HybridSTAFLCController(
            STAController(),
            IntervalType2FLC(config="Taguchi", mf_type=mf_type or "gaussian"),
        )
    elif key in ("7", "PROPOSED", "PROPOSED-AGS-IT2FLC"):
        return "Proposed-AGS-IT2FLC", ProposedAGSIT2FLC()
    else:
        raise ValueError(f"Unknown controller selection: {name}")


def run_demo_simulation(
    angles_deg: Tuple[float, float, float],
    noise_deg: float,
    dist_amp: float,
    t_sim: float,
    controllers_dict: Dict[str, object],
    save_path: Path,
    show_gui: bool = False,
) -> pd.DataFrame:
    """Run live simulation for selected controllers, print metrics table, and save comparison plot."""
    env = SpacecraftEnv(
        dt=0.05,
        t_sim=t_sim,
        eta_init_deg=angles_deg,
        sensor_noise_std_deg=noise_deg,
        sensor_noise_rate_std_deg=noise_deg * 1.5,
        seed=42,
    )

    dist_fn = None
    if dist_amp > 0.0:
        dist_fn = lambda t_val: SpacecraftEnv.command_disturbance_torque(t_val, dist_amp)

    rows = []
    logs = {}
    for name, ctrl in controllers_dict.items():
        sim_data = env.simulate(ctrl, disturbance_fn=dist_fn)
        logs[name] = sim_data
        m = evaluate_simulation(sim_data)
        rows.append({
            "Controller": name,
            "ISE": round(m["ISE"], 3),
            "ITSE": round(m["ITSE"], 3),
            "Control_Effort_||E||": round(m["Control_Effort"], 4),
            "J_Index": round(m["J_Index"], 4),
            "Settling_Time_s": round(m["Settling_Time_s"], 2),
            "Chattering_TV": round(m["Chattering_TV"], 3),
            "Mean_Cycle_ms": round(m["Mean_Cycle_ms"], 4),
        })

    df_res = pd.DataFrame(rows)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), dpi=200)
    fig.suptitle(
        f"Live Defense Demo: η(0) = {list(angles_deg)} deg | Noise σ = {noise_deg}° | Disturbance A = {dist_amp} N·m",
        fontweight="bold",
        fontsize=12,
    )

    for name, log in logs.items():
        t = log["time"]
        lw = 2.4 if "Proposed" in name else 1.7
        ls = "-" if "Proposed" in name else "--"
        axes[0, 0].plot(t, log["eta_deg"][:, 0], lw=lw, ls=ls, label=f"{name} (φ)")
        axes[0, 1].plot(t, log["eta_deg"][:, 1], lw=lw, ls=ls, label=f"{name} (θ)")
        axes[1, 0].plot(t, log["eta_deg"][:, 2], lw=lw, ls=ls, label=f"{name} (ψ)")
        axes[1, 1].plot(t, np.linalg.norm(log["eta_deg"], axis=1), lw=lw, ls=ls, label=name)

    axes[0, 0].set_title("Roll Angle φ(t) [deg]", fontweight="bold")
    axes[0, 1].set_title("Pitch Angle θ(t) [deg]", fontweight="bold")
    axes[1, 0].set_title("Yaw Angle ψ(t) [deg]", fontweight="bold")
    axes[1, 1].set_title("Attitude Error Norm ||e(t)|| [deg]", fontweight="bold")
    axes[1, 1].axhline(2.0, color="black", ls=":", lw=1.0, label="±2° Settling Threshold")

    for ax in axes.ravel():
        ax.set_xlabel("Time (s)")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="upper right", fontsize=8.5)

    fig.tight_layout(rect=[0, 0.01, 1, 0.95])
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200)
    if show_gui:
        plt.show()
    plt.close(fig)

    return df_res


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive Live Defense Demo for Spacecraft Attitude Control (Fuzzy Logic Project)"
    )
    parser.add_argument("--controller", type=str, default=None, help="Controller name or ID (1-7)")
    parser.add_argument("--compare-all", action="store_true", help="Compare STA, T1-GA, T2-GA, and Proposed-AGS-IT2FLC")
    parser.add_argument("--angles", type=float, nargs=3, default=None, help="Initial [phi, theta, psi] in degrees")
    parser.add_argument("--noise", type=float, default=0.0, help="Gaussian attitude sensor noise std in degrees")
    parser.add_argument("--disturbance", type=float, default=0.0, help="External disturbance torque amplitude A (N*m)")
    parser.add_argument("--mf", type=str, default=None, choices=["gaussian", "triangular", "trapezoidal"], help="MF type")
    parser.add_argument("--defuzz", type=str, default="centroid", choices=["centroid", "bisector", "mom", "som"])
    parser.add_argument("--duration", type=float, default=60.0, help="Simulation duration in seconds")
    parser.add_argument("--gui", action="store_true", help="Open interactive matplotlib window")
    args = parser.parse_args()

    print("=" * 85)
    print(" LIVE PRESENTATION & ORAL DEFENSE DEMO — 3-DoF SPACECRAFT ATTITUDE CONTROL")
    print("=" * 85)

    if args.angles is None and args.controller is None and not args.compare_all:
        print("\nAvailable Controllers:")
        print("  [1] STA                     (Baseline Super-Twisting SMC)")
        print("  [2] ASTA                    (Adaptive Super-Twisting SMC)")
        print("  [3] STA-Fuzzy-Type1-GA      (Hybrid Type-1 FLC, L=3, 27 rules)")
        print("  [4] STA-Fuzzy-Type1-Taguchi (Hybrid Type-1 FLC, L=7, 147 rules)")
        print("  [5] STA-Fuzzy-Type2-GA      (Hybrid Interval Type-2 FLC, L=3, KM Type-Reduction)")
        print("  [6] STA-Fuzzy-Type2-Taguchi (Hybrid Interval Type-2 FLC, L=7, KM Type-Reduction)")
        print("  [7] Proposed-AGS-IT2FLC     (Proposed Adaptive Gain-Scheduled Hybrid STA-IT2FLC)")
        print("  [8] Compare Key Controllers (STA vs. T1-GA vs. T2-GA vs. Proposed-AGS-IT2FLC)")

        try:
            choice = input("\nSelect controller [1-8, default=8]: ").strip() or "8"
            ang_str = input("Enter initial Euler angles 'phi theta psi' in deg [default: 40 -30 80]: ").strip()
            angles = tuple(float(x) for x in ang_str.split()) if ang_str else (40.0, -30.0, 80.0)
            noise_str = input("Enter sensor noise std in deg [default: 0.0]: ").strip()
            noise_deg = float(noise_str) if noise_str else 0.0
            dist_str = input("Enter external disturbance amplitude A in N*m [default: 0.0]: ").strip()
            dist_amp = float(dist_str) if dist_str else 0.0
        except EOFError:
            choice = "8"
            angles = (40.0, -30.0, 80.0)
            noise_deg = 0.0
            dist_amp = 0.0
    else:
        choice = "8" if args.compare_all else (args.controller or "7")
        angles = tuple(args.angles) if args.angles is not None else (40.0, -30.0, 80.0)
        noise_deg = args.noise
        dist_amp = args.disturbance

    if choice == "8" or args.compare_all:
        controllers = {
            "STA": STAController(),
            "STA-Fuzzy-Type1-GA": HybridSTAFLCController(STAController(), Type1FLC("GA", mf_type=args.mf, defuzz_method=args.defuzz)),
            "STA-Fuzzy-Type2-GA": HybridSTAFLCController(STAController(), IntervalType2FLC("GA", mf_type=args.mf)),
            "Proposed-AGS-IT2FLC": ProposedAGSIT2FLC(),
        }
    else:
        c_name, c_obj = build_controller(choice, mf_type=args.mf, defuzz_method=args.defuzz)
        controllers = {c_name: c_obj}

    out_fig = Path(__file__).resolve().parent / "results" / "figures" / "live_demo_result.png"
    df_demo = run_demo_simulation(
        angles_deg=angles,
        noise_deg=noise_deg,
        dist_amp=dist_amp,
        t_sim=args.duration,
        controllers_dict=controllers,
        save_path=out_fig,
        show_gui=args.gui,
    )

    print("\n" + "-" * 85)
    print(f" SIMULATION RESULTS (Initial Angles={list(angles)} deg, Noise={noise_deg} deg, Dist={dist_amp} N*m)")
    print("-" * 85)
    print(dataframe_to_markdown(df_demo))
    print("\n[OK] Saved live demo plot to: results/figures/live_demo_result.png")


if __name__ == "__main__":
    main()
