#!/usr/bin/env python3
"""
MyceSoil — Tissue volume fraction feasibility estimate
==========================================================
Uses published hyphal density data to bound achievable f_tiss
in the scaffold sensing volume.

Required f_tiss threshold is 0.15 (all three analyte targets, Kalman-filtered)
from composite_fim_differential.py with proper Riccati Kalman.
The old threshold of 0.50 was based on a fixed ×4 approximation and is obsolete.
"""
import numpy as np

HYPHA_DIAMETER_UM  = 4.0
HYPHA_RADIUS_M     = HYPHA_DIAMETER_UM * 1e-6 / 2.0
HYPHA_VOLUME_PER_M = np.pi * HYPHA_RADIUS_M**2

HYPHAL_LENGTH_SOIL_m_per_g = [50, 200, 1000]
ENRICHMENT_FACTOR     = 10
SCAFFOLD_VOLUME_CM3   = 10.0
SCAFFOLD_VOLUME_M3    = SCAFFOLD_VOLUME_CM3 * 1e-6
SOIL_BULK_DENSITY     = 1.2
SOIL_MASS_G           = SCAFFOLD_VOLUME_CM3 * SOIL_BULK_DENSITY

REQUIRED_F_TISS       = 0.15   # from proper Riccati Kalman, composite_fim_differential.py

def estimate_f_tiss(m_per_g, enrichment=1.0):
    vol = m_per_g * SOIL_MASS_G * enrichment * HYPHA_VOLUME_PER_M
    return vol / SCAFFOLD_VOLUME_M3

def main():
    print("MyceSoil — f_tiss feasibility from hyphal density")
    print(f"Required threshold: f_tiss ≥ {REQUIRED_F_TISS} (all targets, Kalman)")
    print("="*62)
    print(f"\n  Hypha diameter: {HYPHA_DIAMETER_UM} µm")
    print(f"  Scaffold volume: {SCAFFOLD_VOLUME_CM3} cm³  |  Soil mass: {SOIL_MASS_G:.1f} g")

    print("\n  Bulk soil (no scaffold enrichment):")
    for lbl, v in zip(["Low 50 m/g", "Med 200 m/g", "High 1000 m/g"],
                      HYPHAL_LENGTH_SOIL_m_per_g):
        f = estimate_f_tiss(v, 1.0)
        ok = "✓" if f >= REQUIRED_F_TISS else "✗"
        print(f"    {lbl:<16}: f_tiss = {f:.4f}  {ok}")

    print(f"\n  With scaffold enrichment (×{ENRICHMENT_FACTOR}):")
    for lbl, v in zip(["Low 50 m/g", "Med 200 m/g", "High 1000 m/g"],
                      HYPHAL_LENGTH_SOIL_m_per_g):
        f = estimate_f_tiss(v, ENRICHMENT_FACTOR)
        ok = "✓ MEETS threshold" if f >= REQUIRED_F_TISS else "✗ below threshold"
        print(f"    {lbl:<16}: f_tiss = {f:.4f}  {ok}")

    print(f"""
  Summary:
  - Without enrichment: f_tiss is 0.001–0.015 — well below threshold
  - With ×10 scaffold enrichment, high hyphal density: f_tiss = 0.15 — at threshold
  - The gap is marginal, not fatal. Whether real deployments hit 0.15 is the
    primary empirical question of Experiment 1.
  - The enrichment factor is an assumed ×10. Real scaffolds may differ.
    Measuring enrichment is a secondary output of Experiment 1.
""")

if __name__ == "__main__":
    main()
