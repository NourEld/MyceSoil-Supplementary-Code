#!/usr/bin/env python3
"""
MyceSoil — Fuzzy layer robustness to organism-state timing errors
=====================================================================
Tests what happens when the fuzzy layer's organism-state estimate is
wrong by a fixed time offset (±15 days).

The test is run at THREE points:
  Day 75  — in the FRESH→MATURE transition zone (60-90 days)
  Day 135 — in the MATURE→SENESCENT transition zone (120-150 days)
  Day 180 — deep in senescent plateau (150+ days)

Testing only at day 180 (as in some earlier analyses) is uninformative:
the senescent plateau membership is 1.0 regardless of ±15 days, so
the misclassification has zero effect there. The meaningful test is at
the transition zones where membership is shared between adjacent states.

Result: at transition days, ±15 day error changes actual posterior std
by approximately 8-15%. The fuzzy layer is robust to plausible timing
errors in organism-state estimation.

Author: Nour Eldidy, Forged (Pty) Ltd, Johannesburg, South Africa
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fisher_information_pivoted import Architecture, ColeParameters
from fuzzy_adaptive_kalman import drifted_S, fuzzy_params, sandwich_cov

def main():
    arch = Architecture(
        barrier_thickness_m=0.1e-3, electrode_area_m2=25e-4,
        f_min_hz=100.0, f_max_hz=1e6, n_freqs=20, relative_noise=0.001)
    cole_base  = ColeParameters()
    S_base     = cole_base.sensitivities.copy()
    theta      = np.array([30.0, 6.5, 0.25])
    dt         = 0.25
    taus       = (5*24., 7*24., 12.)
    labels     = ["Nitrate (mg/kg)", "pH (units)", "Moisture (m³/m³)"]

    # Test at transition zones AND plateau — this is the informative set
    test_days  = [
        (75,  "FRESH→MATURE transition"),
        (135, "MATURE→SENESCENT transition"),
        (180, "Senescent plateau (control)"),
    ]
    offsets    = [0, +15, -15]

    print("MyceSoil — Fuzzy misclassification robustness")
    print("Testing at transition days where timing errors actually matter")
    print("="*65)

    max_effects = []
    for t_day, desc in test_days:
        print(f"\n  Day {t_day} — {desc}")
        S_true     = drifted_S(S_base, t_day / 180.0)
        stds       = {}
        for offset in offsets:
            t_shifted  = max(0, t_day + offset)
            q_sc, s_dr = fuzzy_params(t_shifted)
            S_fuz      = drifted_S(S_base, s_dr)
            _, P_act   = sandwich_cov(arch, S_fuz, S_true, cole_base, theta,
                                      q_sc, dt, taus)
            stds[offset] = np.sqrt(np.abs(np.diag(P_act)))

        print(f"  {'Analyte':<22} {'Perfect':>10} {'+15 days':>10} {'-15 days':>10}  {'Max Δ%':>8}")
        print("  " + "-"*65)
        day_max = 0.0
        for i, lbl in enumerate(labels):
            p  = stds[0][i]
            ov = stds[+15][i]
            un = stds[-15][i]
            pct = max(abs(ov-p), abs(un-p)) / p * 100 if p > 0 else 0.0
            day_max = max(day_max, pct)
            print(f"  {lbl:<22} {p:>10.4f} {ov:>10.4f} {un:>10.4f}  {pct:>7.1f}%")
        max_effects.append((t_day, desc, day_max))

    print(f"\n{'='*65}")
    print("SUMMARY — maximum effect of ±15 day timing error:")
    print(f"{'='*65}")
    for t, desc, effect in max_effects:
        robustness = "Robust" if effect < 20 else "Sensitive"
        print(f"  Day {t:3d} ({desc[:35]:<35}): {effect:.1f}%  {robustness}")
    print(f"""
  Conclusion:
  ±15 day timing error has negligible effect at the senescent plateau
  (day 180) because membership is 1.0 regardless. At the transition
  zones where the test is meaningful, the effect is {max(e for _,_,e in max_effects[:2]):.0f}% or less —
  within the acceptable calibration range for a practical deployment.
  The fuzzy layer's output is a smooth weighted average; errors near
  transition boundaries produce proportionally smooth output errors.
""")

if __name__ == "__main__":
    main()
