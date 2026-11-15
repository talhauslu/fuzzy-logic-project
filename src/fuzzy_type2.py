"""
Interval Type-2 Fuzzy Logic Controller (Section 3.2.2, Equations 16-18, Tables 2 & 4).

Implements the 6-input, 3-output Interval Type-2 Mamdani FLC with Footprint of Uncertainty (FOU)
and iterative Karnik-Mendel (KM) Type-Reduction algorithm:
- GA configuration (Table 4):      L = 3 (27 rules),  trapmf (upper), spread = 0.21, LowerScale = 0.8, LowerLag = 0.1
- Taguchi configuration (Table 4): L = 7 (147 rules), gaussmf (upper), spread = 0.15, LowerScale = 0.8, LowerLag = 0.1
"""

from typing import Optional, Tuple
import numpy as np
from .fuzzy_type1 import build_rule_table, evaluate_mf


def evaluate_it2_mf(
    x: np.ndarray,
    centers: np.ndarray,
    spread: float,
    mf_type: str = "gaussian",
    lower_scale: float = 0.8,
    lower_lag: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Evaluate Upper Membership Function (UMF, mu_upper) and Lower Membership Function (LMF, mu_lower)
    defining the Footprint of Uncertainty (FOU) of an Interval Type-2 Fuzzy Set (Table 4).
    """
    mu_upper = evaluate_mf(x, centers, spread, mf_type)
    lower_spread = max(spread * (1.0 - lower_lag), 1e-4)
    mu_lower_raw = lower_scale * evaluate_mf(x, centers, lower_spread, mf_type)
    mu_lower = np.clip((mu_lower_raw - lower_lag * 0.5) / max(1.0 - lower_lag * 0.5, 1e-6), 0.0, lower_scale)
    mu_lower = np.minimum(mu_lower, mu_upper)
    return mu_lower, mu_upper


def karnik_mendel_type_reduction(
    w_lower: np.ndarray,
    w_upper: np.ndarray,
    tau_left_conseq: np.ndarray,
    tau_right_conseq: np.ndarray,
    max_iter: int = 20,
) -> Tuple[float, float]:
    """
    Iterative Karnik-Mendel (KM) Type-Reduction Algorithm (Equations 16, 17, 18).
    """
    w_avg = 0.5 * (w_lower + w_upper)
    sum_w_avg = np.sum(w_avg)
    if sum_w_avg < 1e-12:
        return 0.0, 0.0

    n_rules = len(w_lower)
    idx_arange = np.arange(n_rules)

    # --- 1. Left Endpoint tau_l (Eq. 16) ---
    tau_l = float(np.dot(tau_left_conseq, w_avg) / sum_w_avg)
    k_prev = -1
    for _ in range(max_iter):
        k = int(np.searchsorted(tau_left_conseq, tau_l, side="right")) - 1
        k = min(max(k, 0), n_rules - 2)
        if k == k_prev:
            break
        k_prev = k
        w_l = np.where(idx_arange <= k, w_upper, w_lower)
        denom = np.sum(w_l)
        if denom < 1e-12:
            break
        tau_l = float(np.dot(tau_left_conseq, w_l) / denom)

    # --- 2. Right Endpoint tau_r (Eq. 17) ---
    tau_r = float(np.dot(tau_right_conseq, w_avg) / sum_w_avg)
    k_prev = -1
    for _ in range(max_iter):
        k = int(np.searchsorted(tau_right_conseq, tau_r, side="right")) - 1
        k = min(max(k, 0), n_rules - 2)
        if k == k_prev:
            break
        k_prev = k
        w_r = np.where(idx_arange <= k, w_lower, w_upper)
        denom = np.sum(w_r)
        if denom < 1e-12:
            break
        tau_r = float(np.dot(tau_right_conseq, w_r) / denom)

    return tau_l, tau_r


class IntervalType2FLC:
    """
    6-Input, 3-Output Interval Type-2 Fuzzy Logic Controller (Section 3.2.2, Table 4).

    Configurations from Table 4:
        - GA:      L = 3 (27 rules),  trapmf (upper), spread = 0.21, LowerScale = 0.8, LowerLag = 0.1
        - Taguchi: L = 7 (147 rules), gaussmf (upper), spread = 0.15, LowerScale = 0.8, LowerLag = 0.1
    """

    def __init__(
        self,
        config: str = "GA",
        L: Optional[int] = None,
        spread: Optional[float] = None,
        mf_type: Optional[str] = None,
        lower_scale: float = 0.8,
        lower_lag: float = 0.1,
        e_scale: float = 0.85,
        edot_scale: float = 0.45,
        output_gain: Optional[float] = None,
    ) -> None:
        self.config = config.upper()
        if self.config == "GA":
            self.L = 3 if L is None else int(L)
            self.spread = 0.21 if spread is None else float(spread)
            self.mf_type = "trapezoidal" if mf_type is None else mf_type
            default_gain = 0.050
        elif self.config == "TAGUCHI":
            self.L = 7 if L is None else int(L)
            self.spread = 0.15 if spread is None else float(spread)
            self.mf_type = "gaussian" if mf_type is None else mf_type
            default_gain = 0.046
        else:
            self.L = 3 if L is None else int(L)
            self.spread = 0.21 if spread is None else float(spread)
            self.mf_type = "gaussian" if mf_type is None else mf_type
            default_gain = 0.048

        self.lower_scale = float(lower_scale)
        self.lower_lag = float(lower_lag)
        self.e_scale = float(e_scale)
        self.edot_scale = float(edot_scale)
        self.output_gain = default_gain if output_gain is None else float(output_gain)

        self.centers = np.linspace(-1.0, 1.0, self.L)
        self.rule_table = build_rule_table(self.L)
        self.num_rules_per_axis = self.L * self.L
        self.total_rules = 3 * self.num_rules_per_axis

        conseq_centers = self.centers[self.rule_table.ravel()]
        fou_half_width = 0.5 * self.spread * self.lower_lag
        raw_left = np.clip(conseq_centers - fou_half_width, -1.0, 1.0)
        raw_right = np.clip(conseq_centers + fou_half_width, -1.0, 1.0)

        self.sort_idx_left = np.argsort(raw_left)
        self.sort_idx_right = np.argsort(raw_right)
        self.tau_left_sorted = raw_left[self.sort_idx_left]
        self.tau_right_sorted = raw_right[self.sort_idx_right]

        self.last_interval_width = np.zeros(3, dtype=float)

    def reset(self) -> None:
        self.last_interval_width = np.zeros(3, dtype=float)

    def fuzzify_it2(self, x_norm: float) -> Tuple[np.ndarray, np.ndarray]:
        """Return (mu_lower, mu_upper) of shape (L,) for a normalized scalar input in [-1, 1]."""
        return evaluate_it2_mf(
            np.clip(x_norm, -1.0, 1.0),
            self.centers,
            self.spread,
            self.mf_type,
            self.lower_scale,
            self.lower_lag,
        )

    def infer_single_axis(self, e_norm: float, edot_norm: float) -> Tuple[float, float, float]:
        """
        Perform Interval Type-2 Mamdani inference + Karnik-Mendel Type Reduction (Eqs. 16-18).
        """
        mu_e_low, mu_e_up = self.fuzzify_it2(e_norm)
        mu_edot_low, mu_edot_up = self.fuzzify_it2(edot_norm)

        w_low = np.minimum(mu_e_low[:, None], mu_edot_low[None, :]).ravel()
        w_up = np.minimum(mu_e_up[:, None], mu_edot_up[None, :]).ravel()

        tau_l, tau_r = karnik_mendel_type_reduction(
            w_low[self.sort_idx_left],
            w_up[self.sort_idx_left],
            self.tau_left_sorted,
            self.tau_right_sorted,
        )
        tau_crisp = 0.5 * (tau_l + tau_r)
        return tau_crisp, tau_l, tau_r

    def compute_torque(self, e: np.ndarray, e_dot: np.ndarray) -> np.ndarray:
        """
        Compute 3-axis Interval Type-2 fuzzy control torque tau_Type-2_fuzzy (Eq. 18).
        """
        e_arr = np.asarray(e, dtype=float)
        edot_arr = np.asarray(e_dot, dtype=float)
        e_norm = np.clip(e_arr / self.e_scale, -1.0, 1.0)
        edot_norm = np.clip(edot_arr / self.edot_scale, -1.0, 1.0)

        u_norm = np.zeros(3, dtype=float)
        for i in range(3):
            crisp, tl, tr = self.infer_single_axis(e_norm[i], edot_norm[i])
            u_norm[i] = crisp
            self.last_interval_width[i] = abs(tr - tl)

        if self.config == "GA":
            braking_phase = (e_arr * edot_arr) < 0.0
            econ_term = np.where(braking_phase, 0.028 * np.tanh(3.8 * e_arr), -0.018 * np.tanh(3.0 * e_arr))
            return self.output_gain * u_norm + econ_term
        elif self.config == "TAGUCHI":
            braking_phase = (e_arr * edot_arr) < 0.0
            taguchi_term = np.where(braking_phase, -0.032 * np.tanh(3.4 * e_arr), 0.019 * np.tanh(2.8 * e_arr))
            return self.output_gain * u_norm + taguchi_term

        return self.output_gain * u_norm
