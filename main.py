"""
Main Execution Script for Spacecraft Attitude Control via Hybrid Super-Twisting & Fuzzy Logic.

Sequentially executes:
1. Baseline Reproduction Experiments (`experiments/baseline/run_baseline.py`)
   - STA, ASTA, STA-Fuzzy-Type1-GA, STA-Fuzzy-Type1-Taguchi, STA-Fuzzy-Type2-GA, STA-Fuzzy-Type2-Taguchi
   - Generates `fig1_membership_functions.png` and `fig2_baseline_comparison.png`
   - Saves `baseline_results.csv` and `baseline_results.md`
2. Proposed Extension, Sensor Noise Robustness & Ablation Experiments (`experiments/proposed/run_proposed.py`)
   - Proposed-AGS-IT2FLC (Adaptive Gain-Scheduled Hybrid STA-IT2FLC with Innovation-Gated Derivative Filter)
   - Sensor Noise (0.2 deg attitude, 0.35 deg/s rate) & External Disturbance Robustness
   - Membership Function Prototype (Gaussian, Triangular, Trapezoidal) & Defuzzification (Centroid, Bisector, MOM, SOM) Ablation
   - Per-Control-Cycle Computational Timing Analysis
   - Generates `fig3_proposed_vs_baseline.png`, `fig4_noise_robustness.png`, `fig5_ablation_study.png`, `fig6_computational_timing.png`
   - Saves `proposed_comparison.csv/.md` and `ablation_and_noise_results.csv/.md`
"""

import time
from pathlib import Path

from experiments.baseline.run_baseline import run_baseline_experiments
from experiments.proposed.run_proposed import run_proposed_experiments
from src.metrics import dataframe_to_markdown


def main() -> None:
    t0 = time.perf_counter()
    print("=" * 95)
    print(" SPACECRAFT ATTITUDE CONTROL: HYBRID SUPER-TWISTING & TYPE-1/TYPE-2 FUZZY LOGIC FRAMEWORK")
    print(" Reference: Sarjoughian et al., Scientific Reports (2026), doi:10.1038/s41598-026-65663-6")
    print("=" * 95)

    print("\n[1/2] Running Baseline Reproduction Experiments (STA, ASTA, T1-GA, T1-Taguchi, T2-GA, T2-Taguchi)...")
    df_baseline, baseline_logs = run_baseline_experiments()
    print("      -> Saved baseline time-series logs to `data/simulation_logs/`")
    print("      -> Saved `results/tables/baseline_results.csv` & `baseline_results.md`")
    print("      -> Saved `results/figures/fig1_membership_functions.png` & `fig2_baseline_comparison.png`")

    print("\n[2/2] Running Proposed Extension (Proposed-AGS-IT2FLC), Noise Robustness & Ablation Studies...")
    df_proposed, df_ablation_noise = run_proposed_experiments(df_baseline, baseline_logs)
    print("      -> Saved `results/tables/proposed_comparison.csv` & `proposed_comparison.md`")
    print("      -> Saved `results/tables/ablation_and_noise_results.csv` & `ablation_and_noise_results.md`")
    print("      -> Saved `fig3_proposed_vs_baseline.png`, `fig4_noise_robustness.png`,")
    print("         `fig5_ablation_study.png`, and `fig6_computational_timing.png`")

    print("\n" + "=" * 95)
    print(" TABLE 1: BASELINE REPRODUCTION RESULTS (COMPARISON WITH PAPER TABLE 5)")
    print("=" * 95)
    print(dataframe_to_markdown(df_baseline))

    print("\n" + "=" * 95)
    print(" TABLE 2: PROPOSED ADAPTIVE GAIN-SCHEDULED HYBRID STA-IT2FLC VS. BASELINES")
    print("=" * 95)
    print(dataframe_to_markdown(df_proposed))

    print("\n" + "=" * 95)
    print(" TABLE 3: SENSOR NOISE ROBUSTNESS & MF / DEFUZZIFICATION ABLATION STUDY")
    print("=" * 95)
    print(dataframe_to_markdown(df_ablation_noise))

    elapsed = time.perf_counter() - t0
    print("\n" + "=" * 95)
    print(f" ALL SIMULATIONS, TABLES, AND 300-DPI FIGURES GENERATED SUCCESSFULLY IN {elapsed:.2f} SECONDS.")
    print("=" * 95)


if __name__ == "__main__":
    main()
