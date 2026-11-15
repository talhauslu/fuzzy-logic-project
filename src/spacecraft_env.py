"""
3-DoF Spacecraft Attitude Control Simulator Environment (Equations 1-10, Table 6).

Models the three-degree-of-freedom air-bearing spacecraft simulator equipped with
three reaction wheels at the Space Laboratory, University of Isfahan, as described in:
Sarjoughian et al., Scientific Reports (2026).
"""

from typing import Callable, Dict, Optional, Tuple
import numpy as np


class SpacecraftEnv:
    """
    3-DoF Nonlinear Spacecraft Attitude Simulator using 4th-order Runge-Kutta (RK4).

    State Variables:
        eta   = [phi, theta, psi]^T           : Euler angles (roll, pitch, yaw) in rad
        omega = [omega_x, omega_y, omega_z]^T : Body angular velocities in rad/s
        h_w   = [h_x, h_y, h_z]^T             : Reaction wheel angular momentum in N*m*s
    """

    # Table 6: Geometric and inertia parameters of the spacecraft attitude simulator
    DEFAULT_INERTIA = np.array([
        [1.814,   -0.1185,  0.02749],
        [-0.1185,  1.735,   0.01697],
        [0.02749,  0.01697, 3.432]
    ], dtype=float)

    DEFAULT_MGR = np.array([0.0, 0.0, 0.326], dtype=float)  # [mgr_x, mgr_y, mgr_z] in N*m
    TAU_MAX = 0.123  # Reaction wheel maximum torque saturation limit (+/- 0.123 N*m)
    HW_MAX = 0.625   # Nominal reaction wheel momentum rating (N*m*s)

    def __init__(
        self,
        dt: float = 0.05,
        t_sim: float = 60.0,
        eta_init_deg: Tuple[float, float, float] = (40.0, -30.0, 80.0),
        omega_init_deg: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        eta_desired_deg: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        tau_max: float = 0.123,
        gravity_residual_scale: float = 0.012,
        sensor_noise_std_deg: float = 0.0,
        sensor_noise_rate_std_deg: float = 0.0,
        seed: int = 42,
    ) -> None:
        """
        Initialize the 3-DoF Spacecraft Attitude Simulator.

        Args:
            dt: Simulation integration and control step size in seconds (default 0.05 s).
            t_sim: Total simulation horizon in seconds (default 60.0 s, matching Fig. 5).
            eta_init_deg: Initial Euler angles [phi, theta, psi] in degrees.
            omega_init_deg: Initial angular velocity [wx, wy, wz] in deg/s.
            eta_desired_deg: Desired reference attitude [phi_d, theta_d, psi_d] in degrees.
            tau_max: Actuator saturation bound in N*m (default 0.123 N*m).
            gravity_residual_scale: Uncompensated residual gravity perturbation scale satisfying
                Assumption 1 & Theorem 1 (K1 > 2*D0) after nominal f2 cancellation in Eq. (13).
            sensor_noise_std_deg: Zero-mean Gaussian attitude sensor noise std in degrees.
            sensor_noise_rate_std_deg: Zero-mean Gaussian rate sensor noise std in deg/s.
            seed: Random seed for reproducible sensor noise.
        """
        self.dt = float(dt)
        self.t_sim = float(t_sim)
        self.I = self.DEFAULT_INERTIA.copy()
        self.I_inv = np.linalg.inv(self.I)
        self.mgr = self.DEFAULT_MGR.copy()
        self.tau_max = float(tau_max)
        self.gravity_residual_scale = float(gravity_residual_scale)

        self.eta_init = np.deg2rad(np.array(eta_init_deg, dtype=float))
        self.omega_init = np.deg2rad(np.array(omega_init_deg, dtype=float))
        self.eta_d = np.deg2rad(np.array(eta_desired_deg, dtype=float))
        self.eta_d_dot = np.zeros(3, dtype=float)
        self.eta_d_ddot = np.zeros(3, dtype=float)

        self.sensor_noise_std = np.deg2rad(sensor_noise_std_deg)
        self.sensor_noise_rate_std = np.deg2rad(sensor_noise_rate_std_deg)
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)

        # Initial wheel angular momentum matching Fig. 5(d) initial offset
        self.hw_init = np.array([0.30, 0.15, 0.25], dtype=float)

        self.t = 0.0
        self.eta = self.eta_init.copy()
        self.omega = self.omega_init.copy()
        self.h_w = self.hw_init.copy()

    def reset(
        self,
        eta_init_deg: Optional[Tuple[float, float, float]] = None,
        omega_init_deg: Optional[Tuple[float, float, float]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Reset simulator state to initial conditions."""
        self.rng = np.random.default_rng(self.seed)
        if eta_init_deg is not None:
            self.eta_init = np.deg2rad(np.array(eta_init_deg, dtype=float))
        if omega_init_deg is not None:
            self.omega_init = np.deg2rad(np.array(omega_init_deg, dtype=float))

        self.t = 0.0
        self.eta = self.eta_init.copy()
        self.omega = self.omega_init.copy()
        self.h_w = self.hw_init.copy()
        return self.get_measurement()

    @staticmethod
    def kinematic_matrix_R(eta: np.ndarray) -> np.ndarray:
        """
        Compute the Euler angle kinematic matrix R(eta) by inverting Eq. (2):
            eta_dot = R(eta) * omega   (Eq. 1)
        With singularity protection on cos(theta).
        """
        phi, theta, _ = eta
        c_phi = np.cos(phi)
        s_phi = np.sin(phi)
        c_theta = np.cos(theta)
        s_theta = np.sin(theta)

        if abs(c_theta) < 1e-6:
            c_theta = np.copysign(1e-6, c_theta if c_theta != 0 else 1.0)
        t_theta = s_theta / c_theta

        R = np.array([
            [1.0, s_phi * t_theta, c_phi * t_theta],
            [0.0, c_phi,          -s_phi],
            [0.0, s_phi / c_theta, c_phi / c_theta]
        ], dtype=float)
        return R

    @staticmethod
    def kinematic_matrix_R_inv(eta: np.ndarray) -> np.ndarray:
        """
        Compute the inverse kinematic matrix R^{-1}(eta) explicitly from Eq. (2):
            omega = R^{-1}(eta) * eta_dot
        """
        phi, theta, _ = eta
        c_phi = np.cos(phi)
        s_phi = np.sin(phi)
        c_theta = np.cos(theta)
        s_theta = np.sin(theta)

        return np.array([
            [1.0,  0.0,   -s_theta],
            [0.0,  c_phi,  s_phi * c_theta],
            [0.0, -s_phi,  c_phi * c_theta]
        ], dtype=float)

    def gravity_disturbance_torque(self, eta: np.ndarray, full_unbalanced: bool = False) -> np.ndarray:
        """
        Compute the center-of-mass gravity imbalance torque d_grav(t) from Eq. (5) and Table 6:
            d(t) = (m * g * r_s) x K
                 = m*g * [ r_y*cos(phi)*cos(theta) - r_z*sin(phi)*cos(theta),
                          -r_x*cos(phi)*cos(theta) - r_z*sin(theta),
                           r_x*sin(phi)*cos(theta) + r_y*sin(theta) ]^T
        """
        phi, theta, _ = eta
        c_phi, s_phi = np.cos(phi), np.sin(phi)
        c_theta, s_theta = np.cos(theta), np.sin(theta)
        mgr_x, mgr_y, mgr_z = self.mgr

        d_raw = np.array([
            mgr_y * c_phi * c_theta - mgr_z * s_phi * c_theta,
            -mgr_x * c_phi * c_theta - mgr_z * s_theta,
            mgr_x * s_phi * c_theta + mgr_y * s_theta
        ], dtype=float)

        if full_unbalanced:
            return d_raw
        return self.gravity_residual_scale * d_raw

    @staticmethod
    def command_disturbance_torque(t: float, amplitude_A: float = 0.0) -> np.ndarray:
        """
        Compute external command disturbance torque from Eq. (55):
            tau_command_disturbance(t) = A * (1 + sin(t / 100)) * [1, 1, 1]^T
        """
        if amplitude_A == 0.0:
            return np.zeros(3, dtype=float)
        val = amplitude_A * (1.0 + np.sin(t / 100.0))
        return np.array([val, val, val], dtype=float)

    def _continuous_dynamics(
        self,
        t: float,
        state: np.ndarray,
        tau_saturated: np.ndarray,
        ext_disturbance: np.ndarray,
    ) -> np.ndarray:
        """
        Evaluate continuous-time state derivatives [eta_dot, omega_dot, hw_dot] (Eqs. 1, 7, 8, 9, 10).
        """
        eta = state[0:3]
        omega = state[3:6]
        h_w = state[6:9]

        # Eq. (1): eta_dot = R(eta) * omega
        R = self.kinematic_matrix_R(eta)
        eta_dot = R @ omega

        # Eq. (5): Gravity disturbance + external torque disturbance
        d_total = self.gravity_disturbance_torque(eta) + ext_disturbance

        # Eq. (8)-(10): I * omega_dot = -omega x (I * omega) + tau + d(t)
        gyroscopic = np.cross(omega, self.I @ omega)
        omega_dot = self.I_inv @ (-gyroscopic + tau_saturated + d_total)

        # Eq. (7): tau = -hw_dot - omega x h_w  =>  hw_dot = -tau - omega x h_w
        hw_dot = -tau_saturated - np.cross(omega, h_w)

        return np.concatenate([eta_dot, omega_dot, hw_dot])

    def get_measurement(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return measured (eta_meas, omega_meas), adding zero-mean Gaussian sensor noise if configured.
        """
        if self.sensor_noise_std > 0.0 or self.sensor_noise_rate_std > 0.0:
            eta_meas = self.eta + self.rng.normal(0.0, self.sensor_noise_std, size=3)
            omega_meas = self.omega + self.rng.normal(0.0, self.sensor_noise_rate_std, size=3)
            return eta_meas, omega_meas
        return self.eta.copy(), self.omega.copy()

    def step(
        self,
        tau_cmd: np.ndarray,
        ext_disturbance: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Advance the spacecraft simulator by one time step dt using 4th-order Runge-Kutta (RK4).
        """
        tau_applied = np.clip(np.asarray(tau_cmd, dtype=float), -self.tau_max, self.tau_max)
        d_ext = np.zeros(3, dtype=float) if ext_disturbance is None else np.asarray(ext_disturbance, dtype=float)

        state = np.concatenate([self.eta, self.omega, self.h_w])
        dt = self.dt
        t = self.t

        k1 = self._continuous_dynamics(t, state, tau_applied, d_ext)
        k2 = self._continuous_dynamics(t + 0.5 * dt, state + 0.5 * dt * k1, tau_applied, d_ext)
        k3 = self._continuous_dynamics(t + 0.5 * dt, state + 0.5 * dt * k2, tau_applied, d_ext)
        k4 = self._continuous_dynamics(t + dt, state + dt * k3, tau_applied, d_ext)

        next_state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        self.eta = next_state[0:3]
        self.omega = next_state[3:6]
        self.h_w = next_state[6:9]
        self.t += dt

        eta_meas, omega_meas = self.get_measurement()
        return eta_meas, omega_meas, tau_applied

    def simulate(
        self,
        controller,
        disturbance_fn: Optional[Callable[[float], np.ndarray]] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Run a complete closed-loop attitude control simulation over [0, t_sim].
        """
        import time as _time

        self.reset()
        if hasattr(controller, "reset"):
            controller.reset()

        n_steps = int(np.floor(self.t_sim / self.dt)) + 1
        t_hist = np.zeros(n_steps, dtype=float)
        eta_hist = np.zeros((n_steps, 3), dtype=float)
        omega_hist = np.zeros((n_steps, 3), dtype=float)
        e_hist = np.zeros((n_steps, 3), dtype=float)
        edot_hist = np.zeros((n_steps, 3), dtype=float)
        tau_hist = np.zeros((n_steps, 3), dtype=float)
        hw_hist = np.zeros((n_steps, 3), dtype=float)
        cycle_times_ms = np.zeros(n_steps, dtype=float)

        eta_meas, omega_meas = self.get_measurement()

        t_start_total = _time.perf_counter()
        for k in range(n_steps):
            t_cur = self.t
            t_hist[k] = t_cur

            eta_hist[k] = self.eta
            omega_hist[k] = self.omega
            e_hist[k] = self.eta - self.eta_d
            edot_hist[k] = self.omega - self.eta_d_dot
            hw_hist[k] = self.h_w

            t0_cycle = _time.perf_counter()
            tau_cmd = controller.compute_torque(eta_meas, omega_meas, self.dt)
            t1_cycle = _time.perf_counter()
            cycle_times_ms[k] = (t1_cycle - t0_cycle) * 1000.0

            tau_applied = np.clip(tau_cmd, -self.tau_max, self.tau_max)
            tau_hist[k] = tau_applied

            if k < n_steps - 1:
                d_ext = disturbance_fn(t_cur) if disturbance_fn is not None else None
                eta_meas, omega_meas, _ = self.step(tau_applied, d_ext)

        total_exec_time_s = _time.perf_counter() - t_start_total

        return {
            "time": t_hist,
            "eta_rad": eta_hist,
            "eta_deg": np.rad2deg(eta_hist),
            "omega_rad": omega_hist,
            "omega_deg": np.rad2deg(omega_hist),
            "e_rad": e_hist,
            "edot_rad": edot_hist,
            "tau": tau_hist,
            "h_w": hw_hist,
            "cycle_times_ms": cycle_times_ms,
            "total_exec_time_s": total_exec_time_s,
        }
