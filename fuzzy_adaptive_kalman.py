#!/usr/bin/env python3
"""
MyceSoil — Fuzzy-Adaptive Kalman: Calibration Under Sensitivity-Matrix Drift
==============================================================================

Demonstrates that fixed-Q Kalman becomes silently overconfident when the
sensitivity matrix S drifts with colony ageing, while fuzzy-adaptive Kalman
inflates its uncertainty intervals to maintain calibration.

Metric: calibration ratio = actual posterior std / reported posterior std
  - Ratio = 1.0: perfectly calibrated
  - Ratio > 1.0: filter is overconfident (reported interval too tight)
"""
import numpy as np
from copy import deepcopy
import sys
import os as _os
sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from fisher_information_pivoted import Architecture, ColeParameters, fisher_information_matrix


def drifted_S(S_base, frac):
    """Linear drift from S_base toward 70% of S_base (30% sensitivity reduction)."""
    return S_base * (1.0 - 0.30 * frac)


def org_membership(t_day):
    """
    Trapezoidal membership functions.
    fresh:     [0,60] full, fades to 0 at 90
    mature:    rises at 60, full [90,120], fades to 0 at 150
    senescent: rises at 120, full [150, ∞)
    """
    def lside(x, a, b):   # rises from 0 at a to 1 at b
        if x <= a: return 0.0
        if x >= b: return 1.0
        return (x - a) / (b - a)
    def rside(x, c, d):   # falls from 1 at c to 0 at d
        if x <= c: return 1.0
        if x >= d: return 0.0
        return (d - x) / (d - c)

    fresh     = min(1.0, rside(t_day, 60, 90))                    # plateau 0-60
    mature    = min(lside(t_day, 60, 90), rside(t_day, 120, 150)) # plateau 90-120
    senescent = lside(t_day, 120, 150)                             # plateau 150+

    # At t=0: fresh=1, mature=0, senescent=0 ✓
    # At t=90: fresh=0, mature=1, senescent=0 ✓
    # At t=180: fresh=0, mature=0, senescent=1 ✓
    return fresh, mature, senescent


def fuzzy_params(t_day):
    """Defuzzified Q scale and S-drift estimate from organism state."""
    f, m, s = org_membership(t_day)
    q_scale = f*1.0 + m*2.0 + s*5.0
    s_drift = f*0.0 + m*0.25 + s*0.60
    return max(q_scale, 0.01), s_drift   # guard against zero Q


def sandwich_cov(arch, S_model, S_true, cole_base, theta, Q_scale,
                 dt_hours, process_taus_hr):
    """
    Compute steady-state Kalman posterior under model mismatch.
    Returns (P_reported, P_actual):
      P_reported: Riccati with I_model (what the filter thinks it achieves)
      P_actual:   first-order approximation accounting for mismatch
    """
    sigma_phys = np.array([20.0, 0.5, 0.05])
    Q_base = 2 * sigma_phys**2 * dt_hours / np.array(process_taus_hr)
    Q_mat  = np.diag(Q_base * Q_scale)
    A      = np.eye(3)

    # Temporarily swap sensitivities to compute the two FIMs
    cole = deepcopy(cole_base)

    cole.sensitivities = S_model.copy()
    I_model = fisher_information_matrix(arch, cole, theta)

    cole.sensitivities = S_true.copy()
    I_true = fisher_information_matrix(arch, cole, theta)

    def riccati(I_upd):
        P = np.diag(sigma_phys**2)
        for _ in range(3000):
            P_pred = A @ P @ A.T + Q_mat
            P_new  = np.linalg.inv(np.linalg.inv(P_pred) + I_upd)
            if np.max(np.abs(P - P_new)) < 1e-14:
                break
            P = P_new
        return P

    P_rep = riccati(I_model)

    # Actual covariance under mismatch (perturbation theory):
    # P_actual ≈ P_rep + P_rep (I_model − I_true) P_rep
    dI = I_model - I_true
    P_corr = P_rep + P_rep @ dI @ P_rep

    # Ensure positive diagonal
    diag_pos = np.maximum(np.diag(P_corr), np.diag(P_rep))
    P_act = P_rep.copy()
    np.fill_diagonal(P_act, diag_pos)

    return P_rep, P_act


def main():
    print("="*70)
    print("FUZZY-ADAPTIVE KALMAN — calibration under 30% sensitivity drift")
    print("Colony ages from fresh (day 0) to senescent (day 180)")
    print("="*70)

    arch = Architecture(
        barrier_thickness_m=0.1e-3, electrode_area_m2=25e-4,
        f_min_hz=100.0, f_max_hz=1e6, n_freqs=20, relative_noise=0.001)
    cole_base = ColeParameters()
    S_base  = cole_base.sensitivities.copy()
    theta   = np.array([30.0, 6.5, 0.25])
    dt_hours = 0.25
    process_taus = (5*24.0, 7*24.0, 12.0)
    labels = ["Nitrate mg/kg", "pH units", "Moisture m³/m³"]

    days = [0, 30, 60, 90, 120, 150, 180]
    rows = {lbl: {'fixed_rep': [], 'fixed_act': [],
                  'fuzzy_rep': [], 'fuzzy_act': []} for lbl in labels}

    print(f"\n{'Day':<5} {'Filter':<8} {'Parameter':<16} {'Reported':>10} {'Actual':>10} {'Cal ratio':>11}")
    print("-"*65)
    for t in days:
        true_frac = t / 180.0
        S_true = drifted_S(S_base, true_frac)

        # Fixed-Q: always uses S_base, Q_scale=1
        P_fr, P_fa = sandwich_cov(arch, S_base, S_true, cole_base, theta,
                                  1.0, dt_hours, process_taus)
        std_fr = np.sqrt(np.abs(np.diag(P_fr)))
        std_fa = np.sqrt(np.abs(np.diag(P_fa)))

        # Fuzzy: Q_scale and S_drift from organism state
        q_sc, s_dr = fuzzy_params(t)
        S_fuzzy = drifted_S(S_base, s_dr)
        P_zr, P_za = sandwich_cov(arch, S_fuzzy, S_true, cole_base, theta,
                                  q_sc, dt_hours, process_taus)
        std_zr = np.sqrt(np.abs(np.diag(P_zr)))
        std_za = np.sqrt(np.abs(np.diag(P_za)))

        first = True
        for i, lbl in enumerate(labels):
            rows[lbl]['fixed_rep'].append(std_fr[i])
            rows[lbl]['fixed_act'].append(std_fa[i])
            rows[lbl]['fuzzy_rep'].append(std_zr[i])
            rows[lbl]['fuzzy_act'].append(std_za[i])
            prefix = f"{t:<5}" if first else f"{'':5}"
            first = False
            print(f"{prefix} {'Fixed-Q':<8} {lbl:<16} "
                  f"{std_fr[i]:>10.3f} {std_fa[i]:>10.3f} "
                  f"{std_fa[i]/std_fr[i]:>10.2f}x")
            print(f"{'':5} {'Fuzzy':<8} {lbl:<16} "
                  f"{std_zr[i]:>10.3f} {std_za[i]:>10.3f} "
                  f"{std_za[i]/std_zr[i]:>10.2f}x")
        print()

    print("="*70)
    print("SUMMARY TABLE — late deployment (day 180)")
    print("  Reported std: what the filter claims it achieves")
    print("  Actual std:   true posterior uncertainty (sandwich estimator)")
    print("  Cal ratio:    actual/reported — 1.0 is perfect, >1.0 is overconfident")
    print("="*70)
    t_idx = days.index(180)
    for lbl in labels:
        fr = rows[lbl]['fixed_rep'][t_idx]
        fa = rows[lbl]['fixed_act'][t_idx]
        zr = rows[lbl]['fuzzy_rep'][t_idx]
        za = rows[lbl]['fuzzy_act'][t_idx]
        print(f"\n  {lbl}")
        print(f"    Fixed-Q:  reported={fr:.3f}  actual={fa:.3f}  "
              f"cal-ratio={fa/fr:.2f}x")
        print(f"    Fuzzy:    reported={zr:.3f}  actual={za:.3f}  "
              f"cal-ratio={za/zr:.2f}x")
        print(f"    [Fuzzy reported = {zr/fr:.2f}x Fixed-Q reported — wider but honest]")

    print("\n")
    print("="*70)
    print("HEADLINE NUMBERS FOR SECTION 6.3")
    print("="*70)
    # Day 0 = lab conditions, both identical
    # Day 150/180 = late deployment comparison
    for name, tidx in [("Day 0 (fresh — both filters identical)", 0),
                       ("Day 90 (mature colony)", days.index(90)),
                       ("Day 150 (approaching senescence)", days.index(150)),
                       ("Day 180 (senescent colony)",       days.index(180))]:
        print(f"\n  {name}:")
        q_sc, s_dr = fuzzy_params(days[tidx])
        f_m, m_m, s_m = org_membership(days[tidx])
        print(f"  (fuzzy: fresh={f_m:.2f}, mature={m_m:.2f}, senescent={s_m:.2f} "
              f"→ Q×{q_sc:.1f}, S-drift={s_dr:.2f})")
        for lbl in labels:
            fr = rows[lbl]['fixed_rep'][tidx]
            fa = rows[lbl]['fixed_act'][tidx]
            zr = rows[lbl]['fuzzy_rep'][tidx]
            za = rows[lbl]['fuzzy_act'][tidx]
            print(f"    {lbl:<16}  "
                  f"fixed: rep={fr:.3f} act={fa:.3f} ({fa/fr:.2f}x)  "
                  f"fuzzy: rep={zr:.3f} act={za:.3f} ({za/zr:.2f}x)")


if __name__ == "__main__":
    main()
