#!/usr/bin/env python3
"""
MyceSoil — Sensitivity matrix robustness analysis
=====================================================
Perturbs each entry of ColeParameters.sensitivities by random factors
and recomputes the FIM condition number and Kalman CRLBs. Reports the
fraction of trials that meet agronomic targets and the distribution of CRLBs.
"""
import numpy as np
import sys, os
from copy import deepcopy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fisher_information_pivoted import (
    Architecture, ColeParameters, fisher_information_matrix, kalman_crlb,
    TARGET_VALUES, ANALYTE_LABELS
)

def perturb_sensitivities(S_base, max_rel_error=0.30, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    return S_base * (1.0 + rng.uniform(-max_rel_error, max_rel_error, S_base.shape))

def main():
    arch = Architecture(
        barrier_thickness_m=0.1e-3, electrode_area_m2=25e-4,
        f_min_hz=100.0, f_max_hz=1e6, n_freqs=20, relative_noise=0.001,
    )
    cole_base = ColeParameters()
    theta_nom = np.array([30.0, 6.5, 0.25])
    n_trials = 1000

    print("MyceSoil — Sensitivity matrix robustness")
    print("How much can S be wrong before the framework breaks?")
    print("="*62)

    for max_err in [0.10, 0.20, 0.30, 0.50]:
        conds, snap_crlb, kf_crlb = [], np.zeros((n_trials,3)), np.zeros((n_trials,3))
        meets_kf = 0
        rng = np.random.default_rng(42)
        for t in range(n_trials):
            S_pert = perturb_sensitivities(cole_base.sensitivities, max_err, rng)
            cole = deepcopy(cole_base)
            cole.sensitivities = S_pert
            try:
                I = fisher_information_matrix(arch, cole, theta_nom)
                cond = np.linalg.cond(I)
                snap = np.sqrt(np.abs(np.diag(np.linalg.inv(I))))
            except Exception:
                cond, snap = np.inf, np.array([np.inf]*3)
            try:
                kf = kalman_crlb(arch, cole, theta_nom, (5*24.,7*24.,12.), 0.25)
            except Exception:
                kf = np.array([np.inf]*3)
            conds.append(cond); snap_crlb[t] = snap; kf_crlb[t] = kf
            if all(kf[i] < TARGET_VALUES[i] for i in range(3)):
                meets_kf += 1

        conds = np.array(conds)
        valid = np.isfinite(conds)
        print(f"\n  S perturbation ±{max_err*100:.0f}%  |  "
              f"Kalman targets met: {meets_kf/n_trials:.0%}")
        print(f"  {'Analyte':<22} {'Median Kalman':>14}  {'95th pct':>10}  {'Meets?':>8}")
        print("  " + "-"*58)
        for i, lbl in enumerate(ANALYTE_LABELS):
            vals = kf_crlb[valid, i]
            med = np.median(vals); wc = np.percentile(vals, 95)
            ok = "✓" if wc < TARGET_VALUES[i] else "✗"
            print(f"  {lbl:<22} {med:>14.4f}  {wc:>10.4f}  {ok:>8}")

if __name__ == "__main__":
    main()
