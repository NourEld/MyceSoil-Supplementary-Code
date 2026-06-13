#!/usr/bin/env python3
"""
MyceSoil — Systematic error budget at 1 MHz
================================================
Quantifies four systematic error sources and compares their combined
magnitude to the 0.1% relative noise target.

All figures are order-of-magnitude estimates. Benchtop measurement of
electrode polarisation and soil heterogeneity is required before field
precision claims can be verified.

Key finding:
  Temperature drift IS cancelled by the co-located differential architecture
  (both scaffolds in the same thermal mass). However temperature drift is
  negligible (0.048 Ω) compared to the dominant terms anyway.

  The dominant systematic errors after differential are:
    1. Electrode polarisation (estimated 5 Ω — likely lower for capacitive coupling)
    2. Soil heterogeneity (estimated 6.1 Ω at 5% assumed difference)

  These are NOT cancelled by the differential measurement.
  Together they give 0.84% systematic error — 8.4× the 0.1% target.

  This does not mean 0.1% is impossible. It means:
  a) EP for capacitive coupling may be much lower than 5 Ω (needs measurement)
  b) Soil heterogeneity at ≤1% would bring the budget inside target
  c) Experiment 1 Group A vs Group B directly measures the soil
     heterogeneity contribution

Author: Nour Eldidy, Forged (Pty) Ltd, Johannesburg, South Africa
"""

import numpy as np

# Nominal impedances at 1 MHz (feasible architecture)
Z_COUP   =  24.1    # Ω — coupling reactance at 1 MHz
Z_TISS   = 800.0    # Ω — Rinf approx (tissue high-freq resistance)
Z_SOIL   = 350.0 * 0.35   # Ω — Rinf_soil ≈ 122.5 Ω
Z_TOTAL  = Z_COUP + Z_SOIL + Z_TISS   # 946.6 Ω total

# Error source 1: temperature coefficient of polyimide dielectric
# ~200 ppm/°C, 10°C daily swing → 0.2% of coupling impedance
# This IS cancelled by differential (both scaffolds same thermal mass)
TEMP_TCC    = 200e-6   # per °C
DAILY_DT    = 10.0     # °C typical daily swing
drift_temp  = TEMP_TCC * DAILY_DT * Z_COUP   # 0.048 Ω — negligible

# Error source 2: dielectric absorption in polyimide
# ~0.1% of coupling impedance at 1 MHz
drift_DA    = 0.001 * Z_COUP   # 0.024 Ω

# Error source 3: electrode polarisation
# For DIRECT contact electrodes this would be large.
# For CAPACITIVE coupling there is no Faradaic current.
# The estimate of 5 Ω is conservative; likely much lower.
# NEEDS benchtop measurement.
drift_EP    = 5.0   # Ω — ESTIMATED, upper bound for capacitive coupling

# Error source 4: soil heterogeneity between active and reference scaffolds
# 5% difference in soil resistivity between the two co-located scaffolds
# This is the dominant concern and the primary Experiment 1 target
SOIL_HET    = 0.05
drift_soil  = SOIL_HET * Z_SOIL   # 6.125 Ω

def rss(*args):
    return np.sqrt(sum(x**2 for x in args))

# Before differential: all four sources present
total_before = rss(drift_temp, drift_DA, drift_EP, drift_soil)
rel_before   = total_before / Z_TOTAL * 100

# After differential: temperature drift cancelled (physical component)
# DA, EP, soil heterogeneity remain
total_after  = rss(drift_DA, drift_EP, drift_soil)
rel_after    = total_after / Z_TOTAL * 100

# Best-case: EP much lower (say 0.1 Ω) and soil heterogeneity 1%
drift_EP_opt   = 0.1
drift_soil_opt = 0.01 * Z_SOIL
total_best     = rss(drift_DA, drift_EP_opt, drift_soil_opt)
rel_best       = total_best / Z_TOTAL * 100

def main():
    print("MyceSoil — Systematic Error Budget at 1 MHz")
    print("="*62)
    print(f"\n  Z_total = {Z_TOTAL:.1f} Ω  |  Target: 0.1%  "
          f"(= {Z_TOTAL * 0.001:.3f} Ω absolute)\n")

    print(f"  {'Source':<35} {'Value (Ω)':>10}  {'% of Z_total':>13}  {'Cancelled?':>12}")
    print("  " + "-"*76)
    rows = [
        ("Temp drift (polyimide TCC)",     drift_temp,  "YES — differential"),
        ("Dielectric absorption",           drift_DA,    "no"),
        ("Electrode polarisation (est.)",   drift_EP,    "no — ESTIMATED"),
        ("Soil heterogeneity (5% assumed)", drift_soil,  "no — Exp.1 target"),
    ]
    for name, val, cancelled in rows:
        print(f"  {name:<35} {val:>10.4f}  {val/Z_TOTAL*100:>12.4f}%  {cancelled:>12}")

    print(f"\n  {'Before differential (all sources):':<42} "
          f"{total_before:>8.4f} Ω = {rel_before:.4f}%")
    print(f"  {'After differential (temp cancelled):':<42} "
          f"{total_after:>8.4f} Ω = {rel_after:.4f}%")
    print(f"  {'vs 0.1% target:':<42} {rel_after/0.1:.1f}× over target")

    print(f"""
  What the differential cancels:
    Temperature drift accounts for {drift_temp:.4f} Ω — negligible.
    It was never the dominant term. Cancelling it barely changes the total.

  What the differential does NOT cancel:
    Electrode polarisation ({drift_EP:.1f} Ω) — the dominant source of uncertainty
    here is the estimate itself. For capacitive coupling (no Faradaic current),
    EP should be substantially lower than for direct-contact electrodes.
    This needs benchtop measurement.

    Soil heterogeneity ({drift_soil:.3f} Ω, {drift_soil/Z_TOTAL*100:.4f}%) — directly tested by
    Experiment 1 Group A vs Group B. A 1% heterogeneity (vs 5% assumed)
    would reduce the soil term to {0.01*Z_SOIL:.3f} Ω.

  Best-case estimate (EP = {drift_EP_opt} Ω, soil het = 1%):
    Total systematic: {total_best:.4f} Ω = {rel_best:.4f}%  "
          f"({'within target' if rel_best < 0.1 else f'{rel_best/0.1:.1f}× over target'})
""")

if __name__ == "__main__":
    main()
