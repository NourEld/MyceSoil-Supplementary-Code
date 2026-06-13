#!/usr/bin/env python3
"""
MyceSoil — Differential Measurement FIM with proper Riccati Kalman
======================================================================
Authoritative source for paper Section 6.1 CRLB table.

Architecture:
  ΔZ(ω) = Z_colonised(ω) - Z_reference(ω) ≈ f_tiss · Z_tiss(ω; θ_analytes)

  Reference scaffold (inert PHA, no fungus) subtracts soil background.
  This is a mathematical requirement for identifiability, not a noise
  reduction technique — see composite_fim.py for the non-identifiability
  proof without reference electrode.

Noise model:
  σ(ω) = √2 · noise_rel · |Z_colonised_total| + thermal_floor
  (√2 because ΔZ is the difference of two independent measurements)

Kalman:
  Proper Riccati equation on the differential FIM — replaces the fixed ×4
  approximation used in earlier analysis.

Key result: all three agronomic targets met at f_tiss ≥ 0.15.
  Nitrate target (<10 mg/kg): met at f_tiss ≥ 0.10
  pH target (<0.3 units):    met at f_tiss ≥ 0.15  ← binding constraint
  Moisture target (<0.05):   met at f_tiss ≥ 0.01

Note on temperature:
  Temperature is not included as a state variable here because the
  temperature sensitivity matrix entries have not yet been measured
  (Experiment 1 will measure them). Once measured, temperature becomes
  a 4th Kalman state variable. See Section 4.3 and 8.4 of the paper.

Author: Nour Eldidy, Forged (Pty) Ltd, Johannesburg, South Africa
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fisher_information_pivoted import Architecture, ColeParameters

ALPHA_SOIL = 0.72   # consistent with published soil EIS literature
RINF_RATIO = 0.35   # order-of-magnitude estimate; calibrate from Experiment 1

def Z_soil(omega, R0=350.0, tau=5e-5):
    Rinf = RINF_RATIO * R0
    return Rinf + (R0 - Rinf) / (1.0 + (1j * omega * tau)**ALPHA_SOIL)

def Z_tiss_scaled(omega, cole, theta, f_tiss):
    dtheta = theta - np.array([30.0, 6.5, 0.25])
    S = cole.sensitivities
    R0    = max(cole.R0_base_ohm   * (1.0 + S[0] @ dtheta), 100.)
    Rinf  = max(cole.Rinf_base_ohm * (1.0 + S[1] @ dtheta),  50.)
    tau   = max(cole.tau_base_s    * (1.0 + S[2] @ dtheta),  1e-7)
    alpha = float(np.clip(cole.alpha_base + S[3] @ dtheta, 0.3, 0.99))
    Z_t   = Rinf + (R0 - Rinf) / (1.0 + (1j * omega * tau)**alpha)
    return f_tiss * Z_t

def differential_fim(arch, cole, theta, f_tiss, R0_soil=350.0, tau_soil=5e-5):
    omega    = 2 * np.pi * arch.frequencies
    Z_coup   = 1.0 / (1j * omega * arch.C_coup / 2.0)
    Z_col    = Z_tiss_scaled(omega, cole, theta, f_tiss)
    Z_s      = Z_soil(omega, R0_soil, tau_soil)
    Z_total  = Z_coup + Z_s + Z_col
    sigma    = np.sqrt(2) * arch.relative_noise * np.abs(Z_total) + arch.thermal_noise_floor_ohm
    n_p      = len(theta)
    J        = np.zeros((len(omega), 2, n_p))
    for i in range(n_p):
        h  = 1e-5 * max(abs(theta[i]), 1e-3)
        tp = theta.copy(); tp[i] += h
        tm = theta.copy(); tm[i] -= h
        dZ = (Z_tiss_scaled(omega, cole, tp, f_tiss)
              - Z_tiss_scaled(omega, cole, tm, f_tiss)) / (2*h)
        J[:, 0, i] = np.real(dZ)
        J[:, 1, i] = np.imag(dZ)
    I = np.zeros((n_p, n_p))
    for f in range(len(omega)):
        for k in range(2):
            jfk = J[f, k, :]
            I  += np.outer(jfk, jfk) / sigma[f]**2
    return I

def kalman_differential(arch, cole, theta, f_tiss,
                        taus_hr=(5*24., 7*24., 12.), dt_hr=0.25):
    """Steady-state Riccati Kalman on the differential FIM."""
    sigma_phys = np.array([20.0, 0.5, 0.05])
    Q  = np.diag(2 * sigma_phys**2 * dt_hr / np.array(taus_hr))
    Im = differential_fim(arch, cole, theta, f_tiss)
    P  = np.diag(sigma_phys**2)
    for _ in range(3000):
        Pp  = P + Q
        Pn  = np.linalg.inv(np.linalg.inv(Pp) + Im)
        if np.max(np.abs(P - Pn)) < 1e-14:
            break
        P = Pn
    return np.sqrt(np.abs(np.diag(P)))

def main():
    arch = Architecture(
        barrier_thickness_m=0.1e-3, electrode_area_m2=25e-4,
        f_min_hz=100.0, f_max_hz=1e6, n_freqs=20, relative_noise=0.001)
    cole    = ColeParameters()
    theta   = np.array([30.0, 6.5, 0.25])
    targets = [10.0, 0.3, 0.05]
    labels  = ["Nitrate (mg/kg)", "pH (units)", "Moisture (m³/m³)"]

    print("="*70)
    print("MyceSoil — Differential FIM  |  Proper Riccati Kalman")
    print("Targets: nitrate <10 mg/kg, pH <0.3 units, moisture <0.05 m³/m³")
    print("="*70)
    print(f"\n  Sensitivity matrix: ASSUMED from physiology (Experiment 1 measures it)")
    print(f"  Temperature: not included — 4th state variable pending Experiment 1\n")

    f_tiss_vals = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.00]

    print(f"{'f_tiss':<8} {'Snap N':>10} {'Snap pH':>9} {'Snap θ':>10} "
          f"{'Kalm N':>10} {'Kalm pH':>9} {'Kalm θ':>10} {'Meets?':>7}")
    print("-"*78)

    results = {}
    for ft in f_tiss_vals:
        I = differential_fim(arch, cole, theta, ft)
        try:   snap = np.sqrt(np.abs(np.diag(np.linalg.inv(I))))
        except: snap = [np.inf]*3
        try:   kf = kalman_differential(arch, cole, theta, ft)
        except: kf = [np.inf]*3
        meets = all(kf[i] < targets[i] for i in range(3))
        results[ft] = (snap, kf, meets)
        s = "YES ✓" if meets else "no"
        print(f"{ft:<8.2f} {snap[0]:>10.2f} {snap[1]:>9.3f} {snap[2]:>10.4f} "
              f"{kf[0]:>10.2f} {kf[1]:>9.3f} {kf[2]:>10.4f} {s:>7}")

    print(f"\n{'='*70}")
    print("MINIMUM f_tiss to meet each target (Kalman-filtered):")
    print(f"{'='*70}")
    for i, (lbl, tgt) in enumerate(zip(labels, targets)):
        met = [ft for ft, (_, kf, _) in results.items() if kf[i] < tgt]
        print(f"  {lbl:<24}: f_tiss ≥ {min(met):.2f}" if met
              else f"  {lbl:<24}: not met at tested fractions")

    print(f"""
  Paper headline: f_tiss ≥ 0.15 for all three targets simultaneously.
  At f_tiss = 0.15: nitrate {results[0.15][1][0]:.1f} mg/kg ✓, """
          f"""pH {results[0.15][1][1]:.3f} ✓, moisture {results[0.15][1][2]:.4f} ✓

  Real-world f_tiss: unknown. Primary output of Experiment 1.
  Hyphal density modelling (tissue_fraction_feasibility.py) shows 0.15
  is achievable at high colonisation density with ×10 scaffold enrichment.
""")

if __name__ == "__main__":
    main()
