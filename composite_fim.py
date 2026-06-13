#!/usr/bin/env python3
"""
MyceSoil — Composite Forward Model FIM
=========================================

Adds soil impedance to the forward model and asks the real question:

  Can tissue parameters [nitrate, pH, moisture] be identified when
  soil background parameters must be simultaneously estimated?

Forward model (first-order geometric approximation):
  Z_total(ω) = Z_coup(ω) + Z_composite(ω)

  Z_composite(ω) = Z_soil(ω) + f_tiss * Z_tiss(ω)

  where f_tiss is the tissue volume fraction (0.01 = sparse, 0.15 = dense scaffold).

  Rationale: soil dominates the bulk path; tissue contribution is
  a perturbation scaled by its volume fraction. This is a linear
  mixing model — valid when tissue and soil are in parallel conduction
  paths of known relative cross-section. A more rigorous model
  (Archie-type effective medium) is left for Experiment 1 calibration.

State vector (6 parameters):
  θ = [nitrate, pH, moisture,   <- target (we want these)
       R_soil_0, τ_soil, f_tiss] <- nuisance (must estimate jointly)

Key output: Schur complement CRLB for [nitrate, pH, moisture]
  conditional on simultaneously estimating the nuisance soil parameters.
  Compared against the isolated-tissue CRLB from V2.
"""

import numpy as np
from dataclasses import dataclass, field
from copy import deepcopy
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fisher_information_pivoted import Architecture, ColeParameters

EPS0 = 8.854e-12

# ------------------------------------------------------------------
# Soil Cole model
# Parameterised by R_soil_0 (low-freq resistance) and τ_soil (dispersion time).
# α_soil fixed at 0.72 (typical agricultural soil; value consistent
# with published soil EIS literature — not from a specific verified citation).
# R_soil_inf is derived as a fixed fraction of R_soil_0.
# ------------------------------------------------------------------

ALPHA_SOIL_FIXED = 0.72  # consistent with published soil EIS literature; not verified by specific citation
RINF_RATIO_SOIL  = 0.35   # Rinf_soil ≈ 0.35 * R0_soil (order-of-magnitude estimate;
                          # to be calibrated from Experiment 1 Group C data)

def soil_impedance(omega, R_soil_0, tau_soil):
    R_soil_inf = RINF_RATIO_SOIL * R_soil_0
    return R_soil_inf + (R_soil_0 - R_soil_inf) / (
        1.0 + (1j * omega * tau_soil)**ALPHA_SOIL_FIXED)


# ------------------------------------------------------------------
# Composite forward model
# ------------------------------------------------------------------

def total_Z_composite(omega, arch, cole, theta_full):
    """
    theta_full = [nitrate, pH, moisture, R_soil_0, tau_soil, f_tiss]
    """
    nitrate, pH, moisture = theta_full[0], theta_full[1], theta_full[2]
    R_soil_0, tau_soil, f_tiss = theta_full[3], theta_full[4], theta_full[5]

    # Tissue impedance (same Cole model as V2)
    theta_analytes = np.array([nitrate, pH, moisture])
    dtheta = theta_analytes - np.array([30.0, 6.5, 0.25])
    S = cole.sensitivities
    R0   = max(cole.R0_base_ohm   * (1.0 + S[0] @ dtheta), 100.)
    Rinf = max(cole.Rinf_base_ohm * (1.0 + S[1] @ dtheta),  50.)
    tau  = max(cole.tau_base_s    * (1.0 + S[2] @ dtheta),  1e-7)
    alpha = float(np.clip(cole.alpha_base + S[3] @ dtheta, 0.3, 0.99))
    Z_tiss = Rinf + (R0 - Rinf) / (1.0 + (1j * omega * tau)**alpha)

    # Soil impedance
    Z_soil = soil_impedance(omega, R_soil_0, tau_soil)

    # Composite: soil background + tissue perturbation
    Z_mix = Z_soil + f_tiss * Z_tiss

    # Coupling barrier
    Z_coup = 1.0 / (1j * omega * arch.C_coup / 2.0)

    return Z_coup + Z_mix


def fim_composite(arch, cole, theta_full):
    """6x6 FIM for the full composite state vector."""
    omega = 2 * np.pi * arch.frequencies
    Z0 = total_Z_composite(omega, arch, cole, theta_full)
    sigma = arch.relative_noise * np.abs(Z0) + arch.thermal_noise_floor_ohm

    n_p = len(theta_full)
    J = np.zeros((len(omega), 2, n_p))
    for i in range(n_p):
        h = 1e-5 * max(abs(theta_full[i]), 1e-4)
        tp = theta_full.copy(); tp[i] += h
        tm = theta_full.copy(); tm[i] -= h
        dZ = (total_Z_composite(omega, arch, cole, tp)
              - total_Z_composite(omega, arch, cole, tm)) / (2*h)
        J[:, 0, i] = np.real(dZ)
        J[:, 1, i] = np.imag(dZ)

    I = np.zeros((n_p, n_p))
    for f in range(len(omega)):
        for k in range(2):
            jfk = J[f, k, :]
            I += np.outer(jfk, jfk) / sigma[f]**2
    return I


def schur_crlb(I_full, n_target=3):
    """
    Schur complement CRLB for the first n_target parameters,
    marginalising over the remaining nuisance parameters.
    Returns sqrt(diag(inv(I_t|n))).
    """
    I_tt = I_full[:n_target, :n_target]
    I_tn = I_full[:n_target, n_target:]
    I_nn = I_full[n_target:, n_target:]
    try:
        I_nn_inv = np.linalg.inv(I_nn)
        I_cond = I_tt - I_tn @ I_nn_inv @ I_tn.T
        crlb = np.sqrt(np.abs(np.diag(np.linalg.inv(I_cond))))
    except np.linalg.LinAlgError:
        crlb = np.array([np.inf, np.inf, np.inf])
    return crlb


# ------------------------------------------------------------------
# Reference: isolated tissue CRLB (from V2, Kalman steady-state)
# ------------------------------------------------------------------
def isolated_snapshot_crlb(arch, cole, theta_analytes):
    """Reproduce V2 isolated-tissue CRLB for comparison."""
    from fisher_information_pivoted import fisher_information_matrix
    I = fisher_information_matrix(arch, cole, theta_analytes)
    return np.sqrt(np.abs(np.diag(np.linalg.inv(I))))


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    arch = Architecture(
        barrier_thickness_m=0.1e-3,
        electrode_area_m2=25e-4,
        f_min_hz=100.0, f_max_hz=1e6,
        n_freqs=20,
        relative_noise=0.001,
    )
    cole = ColeParameters()

    # Nominal soil parameters at field capacity in Limpopo ferralsol
    # R_soil_0 ≈ 350 Ω at θ_water=0.25 (Amente et al. 2000, African soil EIS)
    # τ_soil ≈ 5×10⁻⁵ s → dispersion peak ~3 kHz (soil α-dispersion)
    R_soil_nom = 350.0
    tau_soil_nom = 5e-5

    theta_analytes_nom = np.array([30.0, 6.5, 0.25])

    labels_target  = ["Nitrate (mg/kg)", "pH (units)", "Moisture (m³/m³)"]
    labels_nuisance = ["R_soil_0 (Ω)", "τ_soil (s)", "f_tiss"]

    # ----- Baseline: isolated tissue (V2 reference) -----
    crlb_iso = isolated_snapshot_crlb(arch, cole, theta_analytes_nom)

    print("="*70)
    print("COMPOSITE FIM — soil + tissue joint inversion")
    print("="*70)
    print(f"\nV2 reference (isolated tissue, no soil term):")
    for i, lbl in enumerate(labels_target):
        print(f"  {lbl:<22}  CRLB = {crlb_iso[i]:.4f}")

    # ----- Sweep: tissue volume fraction f_tiss -----
    print(f"\n{'='*70}")
    print("SCHUR COMPLEMENT CRLB — joint estimation, varying f_tiss")
    print("(f_tiss = tissue volume fraction within electrode sensing volume)")
    print(f"{'='*70}")

    f_tiss_values = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30]

    print(f"\n{'f_tiss':<10} {'Full 6×6 cond':>16} "
          f"{'Nitrate':>12} {'pH':>10} {'Moisture':>12} "
          f"{'vs V2 nitrate':>14}")
    print("-"*80)

    results = {}
    for f_tiss in f_tiss_values:
        theta_full = np.array([30.0, 6.5, 0.25, R_soil_nom, tau_soil_nom, f_tiss])
        I_full = fim_composite(arch, cole, theta_full)
        cond_full = np.linalg.cond(I_full)
        crlb_sch = schur_crlb(I_full, n_target=3)
        ratio_n = crlb_sch[0] / crlb_iso[0]
        results[f_tiss] = {'cond': cond_full, 'crlb': crlb_sch}
        print(f"{f_tiss:<10.3f} {cond_full:>16.2e} "
              f"{crlb_sch[0]:>12.2f} {crlb_sch[1]:>10.3f} {crlb_sch[2]:>12.4f} "
              f"{ratio_n:>12.1f}x worse")

    # ----- Spectral signature overlap analysis -----
    print(f"\n{'='*70}")
    print("SPECTRAL SIGNATURE ANALYSIS — soil vs tissue contribution")
    print("at f_tiss=0.10 (representative dense scaffold)")
    print(f"{'='*70}")

    omega = 2 * np.pi * arch.frequencies
    theta_full_ref = np.array([30.0, 6.5, 0.25, R_soil_nom, tau_soil_nom, 0.10])
    Z_tot   = total_Z_composite(omega, arch, cole, theta_full_ref)
    Z_soil_only  = soil_impedance(omega, R_soil_nom, tau_soil_nom)
    f_tiss_ref = 0.10

    # Tissue Cole at baseline
    Z_tiss_scaled = []
    for i, w in enumerate(omega):
        Z_tiss_raw = (cole.Rinf_base_ohm +
                      (cole.R0_base_ohm - cole.Rinf_base_ohm) /
                      (1.0 + (1j * w * cole.tau_base_s)**cole.alpha_base))
        Z_tiss_scaled.append(f_tiss_ref * Z_tiss_raw)
    Z_tiss_scaled = np.array(Z_tiss_scaled)

    print(f"\n  {'Freq (Hz)':>12} {'|Z_soil|':>12} {'|Z_tiss×f|':>14} {'Tissue %':>10}")
    print("  " + "-"*52)
    for j in range(0, len(omega), 3):
        f_hz = arch.frequencies[j]
        zs = abs(Z_soil_only[j])
        zt = abs(Z_tiss_scaled[j])
        pct = 100 * zt / (zs + zt)
        print(f"  {f_hz:>12.0f} {zs:>12.1f} {zt:>14.1f} {pct:>9.2f}%")

    # ----- High-frequency-only FIM (50 kHz - 1 MHz) -----
    print(f"\n{'='*70}")
    print("HIGH-FREQUENCY SUBSET (50 kHz – 1 MHz only) — does isolation improve?")
    print(f"{'='*70}")

    arch_hf = Architecture(
        barrier_thickness_m=0.1e-3,
        electrode_area_m2=25e-4,
        f_min_hz=50e3, f_max_hz=1e6,
        n_freqs=10,
        relative_noise=0.001,
    )

    print(f"\n{'f_tiss':<10} {'Cond(HF)':>14} "
          f"{'Nitrate':>12} {'pH':>10} {'Moisture':>12}")
    print("-"*56)
    for f_tiss in [0.05, 0.10, 0.15, 0.20]:
        theta_full = np.array([30.0, 6.5, 0.25, R_soil_nom, tau_soil_nom, f_tiss])
        I_hf = fim_composite(arch_hf, cole, theta_full)
        cond_hf = np.linalg.cond(I_hf)
        crlb_hf = schur_crlb(I_hf, n_target=3)
        print(f"{f_tiss:<10.3f} {cond_hf:>14.2e} "
              f"{crlb_hf[0]:>12.2f} {crlb_hf[1]:>10.3f} {crlb_hf[2]:>12.4f}")

    # ----- Key finding summary -----
    print(f"\n{'='*70}")
    print("SUMMARY — identifiability threshold")
    print(f"{'='*70}")

    agronomic = [10.0, 0.3, 0.05]
    print(f"\n  Agronomic targets: nitrate <{agronomic[0]} mg/kg, "
          f"pH <{agronomic[1]}, moisture <{agronomic[2]} m³/m³")
    print(f"\n  f_tiss  Meets all targets (snapshot)")
    print("  " + "-"*40)
    for f_tiss, r in results.items():
        meets = all(r['crlb'][i] < agronomic[i] for i in range(3))
        status = "YES" if meets else "no "
        print(f"  {f_tiss:.3f}   {status}   "
              f"(nitrate: {r['crlb'][0]:.1f} mg/kg)")

    print(f"\n  V2 isolated tissue (no soil term): "
          f"nitrate={crlb_iso[0]:.2f}, pH={crlb_iso[1]:.3f}, moisture={crlb_iso[2]:.4f}")
    print(f"\n  The f_tiss threshold is the minimum colonisation density required")
    print(f"  for identifiable inversion. This is the primary Experiment 1 target.")


if __name__ == "__main__":
    main()
