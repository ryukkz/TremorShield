"""
Synthetic tremor model — the mathematical core of TREMORSHIELD.

--------------------------------------------------------------------------
SOURCE
--------------------------------------------------------------------------
    Kulkarni, S. R., Accoto, D., & Campolo, D. (2024). Viscous damping of
    tremor using a wearable robot with an optimized mechanical
    metamaterial. Wearable Technologies, 5, e20.
    https://doi.org/10.1017/wtc.2024.15

That paper's "Emulated Motion Excitation" (EME) injects a tremulous wrist
angular acceleration at frequencies of 4, 6, 8 and 10 Hz:

    q_ddot_tr(t) = a_tr * r(t) * sin(2*pi*f_tr*t)

  - a_tr = 13 rad/s^2. The paper attributes this specific amplitude value
    to: Rocon, E., Ruiz, A. F., Pons, J. L., Belda-Lois, J. M., &
    Sanchez-Lacuesta, J. J. (2004), characterising pathological tremor in
    the human wrist. This is an ANGULAR ACCELERATION of a wrist joint
    model, not a screen-pixel quantity, so it is deliberately NOT reused
    as a pixel amplitude here.
  - r(t) is "a stochastic value between 0 and 1 generated at each time
    step ... to model the stochastic nature of tremor", citing:
    Randall, R. B. (1973), and Gantert, C., Honerkamp, J., & Timmer, J.
    (1992).
  - The frequency set {4, 6, 8, 10} Hz is taken directly from the paper's
    experimental excitation frequencies.

--------------------------------------------------------------------------
WHAT WE KEEP vs. WHAT WE RE-PARAMETERISE
--------------------------------------------------------------------------
KEEP:   the sinusoid-with-stochastic-envelope structure, the four
        excitation frequencies {4, 6, 8, 10} Hz, and the idea that r(t)
        should vary smoothly rather than jump white-noise-like sample to
        sample.
CHANGE: amplitude is re-parameterised from an angular acceleration
        (rad/s^2) to a configurable SPATIAL pixel amplitude A, applied as
        a 2-D displacement transverse to the instantaneous direction of
        travel. This is an engineering simulation parameter that requires
        its own validation — it is NOT a claim that A pixels is clinically
        equivalent to 13 rad/s^2 at the wrist.

--------------------------------------------------------------------------
ADAPTED 2-D SPATIAL MODEL IMPLEMENTED HERE
--------------------------------------------------------------------------
    p_obs(t) = p_gt(t) + A * r(t) * sin(2*pi*f*t + phi) * n(t)

    p_gt(t) : ground-truth (clean) cursor position [x0(t), y0(t)], taken
              directly from the CSV
    A       : configurable pixel amplitude (one of cfg.amplitudes_px)
    r(t)    : smoothed stochastic envelope in [OU_MIN, OU_MAX], generated
              as a mean-reverting (Ornstein-Uhlenbeck-style) process
              integrated over the ACTUAL (non-uniform) dt of the trial —
              stochastic (Randall, 1973; Gantert et al., 1992) but not a
              discontinuous per-sample jump (Kulkarni et al., 2024)
    f       : tremor frequency, f in {4, 6, 8, 10} Hz
    phi     : random phase per trial, phi ~ U(0, 2*pi)
    n(t)    : unit normal to the local velocity direction of p_gt(t):
                  n(t) = [-vy(t), vx(t)] / sqrt(vx(t)^2 + vy(t)^2)
              i.e. tremor is injected transverse to the intended motion
              (around the path), not as independent isotropic noise on
              x and y.

VALIDATION NOTE (see src/validation.py): when checking that a generated
f Hz signal shows a spectral peak near f Hz, use the SIGNED residual
(observed - ground_truth, per axis), never sqrt(res_x^2 + res_y^2). The
magnitude is a full-wave-rectified sinusoid and rectification doubles the
apparent frequency (|sin(2*pi*f*t)| has fundamental frequency 2f, not f).
"""

from typing import Tuple

import numpy as np
import pandas as pd

from .config import EPS_DT, EPS_SPEED, OU_TAU_SEC, OU_SIGMA, OU_LEVEL, OU_MIN, OU_MAX


def generate_tremor(dt: np.ndarray, freq: float, amplitude_px: float,
                     phase: float, t_local: np.ndarray,
                     rng: np.random.Generator) -> np.ndarray:
    """Return the scalar tremor displacement s(t) for one trial:

        s(t) = amplitude_px * r(t) * sin(2*pi*freq*t_local + phase)

    r(t) is a mean-reverting stochastic process (Ornstein-Uhlenbeck-style),
    integrated over the ACTUAL non-uniform dt of the trial:

        r[i] = r[i-1] + theta*(level - r[i-1])*dt[i] + sigma*sqrt(dt[i])*N(0,1)
        r[i] = clip(r[i], OU_MIN, OU_MAX)
    """
    n = len(dt)
    r = np.empty(n, dtype=float)
    r[0] = np.clip(rng.uniform(0.3, 0.9), OU_MIN, OU_MAX)
    theta = 1.0 / OU_TAU_SEC
    for i in range(1, n):
        dt_i = max(dt[i], EPS_DT)
        drift = theta * (OU_LEVEL - r[i - 1]) * dt_i
        diffusion = OU_SIGMA * np.sqrt(dt_i) * rng.normal()
        r[i] = np.clip(r[i - 1] + drift + diffusion, OU_MIN, OU_MAX)

    s = amplitude_px * r * np.sin(2 * np.pi * freq * t_local + phase)
    return s


def _unit_normals(x: np.ndarray, y: np.ndarray, dt: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Unit normal n(t) = [-vy, vx] / |v| to the clean trajectory, with a
    safe fallback for near-zero velocity samples (forward-fill the last
    valid direction, then back-fill any leading gap, then default to
    (0, 1) if the whole trial is stationary).

    NOTE: computed OFFLINE over the whole trial (uses forward *and*
    backward fill) — appropriate for generating a synthetic dataset but
    not causal, so do not reuse this as-is in a real-time system.
    """
    n = len(x)
    vx = np.zeros(n)
    vy = np.zeros(n)
    dt_safe = np.where(dt > EPS_DT, dt, EPS_DT)
    vx[1:] = (x[1:] - x[:-1]) / dt_safe[1:]
    vy[1:] = (y[1:] - y[:-1]) / dt_safe[1:]

    speed = np.sqrt(vx ** 2 + vy ** 2)
    valid = speed > EPS_SPEED

    nx = np.full(n, np.nan)
    ny = np.full(n, np.nan)
    nx[valid] = -vy[valid] / speed[valid]
    ny[valid] = vx[valid] / speed[valid]

    nx_s = pd.Series(nx).ffill().bfill()
    ny_s = pd.Series(ny).ffill().bfill()
    nx = nx_s.to_numpy().copy()  # .copy(): to_numpy() can return a read-only view
    ny = ny_s.to_numpy().copy()

    still_nan = np.isnan(nx)
    if still_nan.any():
        nx[still_nan] = 0.0
        ny[still_nan] = 1.0

    return nx, ny
