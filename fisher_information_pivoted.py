#!/usr/bin/env python3
"""
MyceSoil — Fisher Information Matrix Analysis
================================================

Core FIM identifiability analysis for the capacitively-coupled
biohybrid soil sensor architecture.

ARCHITECTURE DEFAULTS are set to the FEASIBLE geometry that produces
the paper's headline numbers. If you change them to the naive geometry
(thick barrier, small electrodes, low frequency, high noise) the viability
report will tell you exactly what is wrong and what to fix.

Author: Nour Eldidy, Forged (Pty) Ltd, Johannesburg, South Africa
"""

import numpy as np
from numpy.linalg import det, cond, inv
import json, os
from dataclasses import dataclass, field
from typing import Dict

# ── Agronometric precision targets ────────────────────────────────────
TARGETS = {
    "nitrate_mg_per_kg":    10.0,
    "pH_units":              0.30,
    "moisture_m3_per_m3":   0.050,
}
ANALYTE_LABELS = ["Nitrate (mg/kg)", "pH (units)", "Moisture (m³/m³)"]
TARGET_VALUES  = [TARGETS["nitrate_mg_per_kg"],
                  TARGETS["pH_units"],
                  TARGETS["moisture_m3_per_m3"]]


# ── Physical constants ─────────────────────────────────────────────────

@dataclass
class Architecture:
    """
    Physical constants of the dielectric-coupled measurement.

    DEFAULTS are the FEASIBLE architecture (paper Section 4.3):
      - 100 µm polyimide coverlay barrier
      - 25 cm² electrode area
      - 100 Hz – 1 MHz frequency range
      - 0.1% relative noise (achievable with lock-in integration ≥1s)

    NAIVE geometry (0.5 mm barrier, 1 cm², 100 kHz, 3% noise) will fail
    the viability report — the coupling reactance buries the tissue signal.
    """
    barrier_thickness_m:  float = 0.1e-3       # 100 µm polyimide coverlay
    electrode_area_m2:    float = 25e-4         # 25 cm²
    epsilon_r_barrier:    float = 3.5           # polyimide relative permittivity
    f_min_hz:             float = 100.0
    f_max_hz:             float = 1_000_000.0   # 1 MHz
    n_freqs:              int   = 20
    relative_noise:       float = 0.001         # 0.1%
    thermal_noise_floor_ohm: float = 10.0
    eps0:                 float = 8.854e-12

    @property
    def C_coup(self) -> float:
        return (self.epsilon_r_barrier * self.eps0 *
                self.electrode_area_m2 / self.barrier_thickness_m)

    @property
    def frequencies(self) -> np.ndarray:
        return np.logspace(np.log10(self.f_min_hz),
                           np.log10(self.f_max_hz), self.n_freqs)


@dataclass
class ColeParameters:
    """
    Cole-dispersion parameters for Trichoderma-colonised scaffold tissue.

    Baseline at centre of physiological range:
      [NO3⁻] = 30 mg/kg, pH = 6.5, θ = 0.25 m³/m³

    *** SENSITIVITY MATRIX IS ASSERTED, NOT MEASURED ***
    These values are physiologically reasoned from Pma1 membrane
    electrophysiology. Measuring them is Experiment 1 of the
    validation roadmap. Every CRLB this script produces is conditional
    on these entries being approximately correct.
    """
    R0_base_ohm:   float = 12_000.0
    Rinf_base_ohm: float = 800.0
    tau_base_s:    float = 1.0e-4
    alpha_base:    float = 0.75

    # Rows: [R0, Rinf, tau, alpha]
    # Cols: [nitrate (mg/kg), pH, moisture (m³/m³)]
    sensitivities: np.ndarray = field(default_factory=lambda: np.array([
        [ -0.008,   0.15,   -0.4  ],
        [ -0.001,   0.02,   -3.5  ],
        [ -0.003,   0.08,   -1.2  ],
        [  0.0005, -0.02,   -0.5  ],
    ]))


# ── Forward model ──────────────────────────────────────────────────────

def total_impedance(omega, cole, arch, theta):
    theta_base = np.array([30.0, 6.5, 0.25])
    dtheta = theta - theta_base
    S = cole.sensitivities
    R0    = max(cole.R0_base_ohm   * (1.0 + S[0] @ dtheta), 100.)
    Rinf  = max(cole.Rinf_base_ohm * (1.0 + S[1] @ dtheta),  50.)
    tau   = max(cole.tau_base_s    * (1.0 + S[2] @ dtheta),  1e-7)
    alpha = float(np.clip(cole.alpha_base + S[3] @ dtheta, 0.3, 0.99))
    Z_tiss = Rinf + (R0 - Rinf) / (1.0 + (1j * omega * tau)**alpha)
    Z_coup = 1.0 / (1j * omega * arch.C_coup / 2.0)
    return Z_coup + Z_tiss


# ── FIM computation ────────────────────────────────────────────────────

def numerical_jacobian(arch, cole, theta, eps=1e-4):
    omega = 2 * np.pi * arch.frequencies
    n_freqs, n_params = len(omega), len(theta)
    J = np.zeros((n_freqs, 2, n_params))
    for i in range(n_params):
        h = eps * max(abs(theta[i]), 1e-6)
        tp = theta.copy(); tp[i] += h
        tm = theta.copy(); tm[i] -= h
        dZ = (total_impedance(omega, cole, arch, tp)
              - total_impedance(omega, cole, arch, tm)) / (2 * h)
        J[:, 0, i] = np.real(dZ)
        J[:, 1, i] = np.imag(dZ)
    return J


def fisher_information_matrix(arch, cole, theta):
    omega = 2 * np.pi * arch.frequencies
    Z_nom = total_impedance(omega, cole, arch, theta)
    sigma = arch.relative_noise * np.abs(Z_nom) + arch.thermal_noise_floor_ohm
    J = numerical_jacobian(arch, cole, theta)
    I = np.zeros((len(theta), len(theta)))
    for f in range(len(omega)):
        for k in range(2):
            J_fk = J[f, k, :]
            I += np.outer(J_fk, J_fk) / sigma[f]**2
    return I


# ── Kalman steady-state CRLB ───────────────────────────────────────────

def kalman_crlb(arch, cole, theta_nom,
                process_taus_hr=(5*24., 7*24., 12.),
                dt_hours=0.25):
    sigma_phys = np.array([20.0, 0.5, 0.05])
    Q_base = 2 * sigma_phys**2 * dt_hours / np.array(process_taus_hr)
    Q_mat  = np.diag(Q_base)
    I_nom  = fisher_information_matrix(arch, cole, theta_nom)
    P = np.diag(sigma_phys**2)
    for _ in range(3000):
        P_pred = P + Q_mat
        P_new  = inv(inv(P_pred) + I_nom)
        if np.max(np.abs(P - P_new)) < 1e-14:
            break
        P = P_new
    return np.sqrt(np.abs(np.diag(P)))


# ── Viability report ───────────────────────────────────────────────────

def status_symbol(crlb, target):
    ratio = crlb / target
    if ratio <= 0.5:   return "✓  PASSES   "
    elif ratio <= 1.0: return "⚠  MARGINAL "
    else:              return "✗  FAILS    "

def viability_report(arch, cole,
                     theta_nom=None,
                     f_tiss=0.50,
                     process_taus_hr=(5*24., 7*24., 12.)):
    """
    Print a full viability report for the given architecture.
    The caller can pass any Architecture() instance — this function
    tells them whether it works and what to fix if not.
    """
    if theta_nom is None:
        theta_nom = np.array([30.0, 6.5, 0.25])

    omega = 2 * np.pi * arch.frequencies

    # Coupling diagnostics
    Z_coup_at_fmin = abs(1.0 / (1j * 2*np.pi*arch.f_min_hz * arch.C_coup/2))
    Z_coup_at_fmax = abs(1.0 / (1j * 2*np.pi*arch.f_max_hz * arch.C_coup/2))
    Z_tiss_approx  = cole.R0_base_ohm  # rough tissue impedance

    line = "─" * 62
    print(f"\n╔{line}╗")
    print(f"║  MyceSoil Architecture Viability Report" + " "*23 + "║")
    print(f"╠{line}╣")
    print(f"║  Barrier:     {arch.barrier_thickness_m*1e6:.0f} µm  |  "
          f"Electrode: {arch.electrode_area_m2*1e4:.0f} cm²  |  "
          f"ε_r: {arch.epsilon_r_barrier:.1f}" + " "*10 + "║")
    print(f"║  Frequency:   {arch.f_min_hz:.0f} Hz – {arch.f_max_hz/1e6:.1f} MHz  |  "
          f"Noise: {arch.relative_noise*100:.2g}%  |  "
          f"f_tiss: {f_tiss:.2f}" + " "*6 + "║")
    fmax_label = f"{arch.f_max_hz/1e6:.1f} MHz" if arch.f_max_hz >= 1e6 else f"{arch.f_max_hz/1e3:.0f} kHz"
    print(f"║  Coupling cap: {arch.C_coup*1e12:.1f} pF  |  "
          f"Z_coup at {arch.f_min_hz:.0f} Hz: {Z_coup_at_fmin/1e3:.1f} kΩ  |  "
          f"at {fmax_label}: {Z_coup_at_fmax:.1f} Ω" + " "*2 + "║")

    # Check if coupling is dominated by barrier
    barrier_dominated = Z_coup_at_fmax > Z_tiss_approx * 10
    if barrier_dominated:
        fl2 = f"{arch.f_max_hz/1e6:.1f} MHz" if arch.f_max_hz >= 1e6 else f"{arch.f_max_hz/1e3:.0f} kHz"
        print(f"║  ⚠  BARRIER DOMINATES: Z_coup({fl2}) = "
              f"{Z_coup_at_fmax:.0f} Ω >> Z_tissue ≈ {Z_tiss_approx:.0f} Ω    ║")

    # Snapshot CRLBs
    try:
        I = fisher_information_matrix(arch, cole, theta_nom)
        cond_n = np.linalg.cond(I)
        snap = np.sqrt(np.abs(np.diag(inv(I))))
    except Exception:
        snap = np.array([np.inf, np.inf, np.inf])
        cond_n = np.inf

    # Kalman CRLBs
    try:
        kf = kalman_crlb(arch, cole, theta_nom, process_taus_hr)
    except Exception:
        kf = np.array([np.inf, np.inf, np.inf])

    # Differential penalty at f_tiss
    # Noise inflated by sqrt(2) / f_tiss relative to isolated tissue
    Z_soil_approx = 350.0   # Ω at field capacity
    Z_total_approx = (Z_coup_at_fmax + Z_soil_approx +
                                 f_tiss * cole.Rinf_base_ohm)
    noise_inflation = (np.sqrt(2) * Z_total_approx) / (f_tiss * cole.Rinf_base_ohm)
    snap_diff = snap * noise_inflation
    kf_diff   = kf   * noise_inflation

    print(f"╠{line}╣")
    print(f"║  {'ISOLATED TISSUE MODEL':^25}  "
          f"{'DIFFERENTIAL (f_tiss='+str(f_tiss)+')':^33}  ║")
    print(f"║  {'Analyte':<18} {'Snap':>8} {'Kalman':>8} {'Status':>12}  "
          f"{'Snap':>8} {'Kalman':>8} {'Status':>10}║")
    print(f"╠{line}╣")

    any_fails_isolated = False
    any_fails_diff = False

    for i, (lbl, tgt) in enumerate(zip(
            ["Nitrate mg/kg", "pH units", "Moisture m³/m³"], TARGET_VALUES)):
        st_iso  = status_symbol(kf[i],      tgt)
        st_diff = status_symbol(kf_diff[i], tgt)
        if "FAILS" in st_iso:  any_fails_isolated = True
        if "FAILS" in st_diff: any_fails_diff = True
        print(f"║  {lbl:<18} {snap[i]:>8.2f} {kf[i]:>8.2f} {st_iso:>12}  "
              f"{snap_diff[i]:>8.2f} {kf_diff[i]:>8.2f} {st_diff:>10}║")

    # Condition number
    cn_label = "OK" if cond_n < 1e8 else ("⚠ ILL-CONDITIONED" if cond_n < 1e15 else "✗ SINGULAR")
    print(f"╠{line}╣")
    print(f"║  FIM condition number: {cond_n:.2e}  [{cn_label}]" + " "*20 + "║")

    # Overall verdict
    print(f"╠{line}╣")
    if not any_fails_isolated and not any_fails_diff:
        verdict = "✓  VIABLE — all targets met (isolated and differential)"
    elif not any_fails_isolated and any_fails_diff:
        verdict = "⚠  CONDITIONALLY VIABLE — meets targets in isolated model only"
        verdict2 = f"   Higher f_tiss or signal amplification needed for differential"
    elif not any_fails_diff:
        verdict = "⚠  VIABLE UNDER DIFFERENTIAL ARCHITECTURE — snapshot alone insufficient"
        verdict2 = None
    else:
        verdict = "✗  NOT VIABLE — architecture changes required"
        verdict2 = None

    print(f"║  {verdict:<60}║")
    if any_fails_diff and any_fails_isolated:
        print(f"║  {'Bottleneck diagnosis below':^60}║")
    print(f"╠{line}╣")

    # Bottleneck diagnosis
    issues = []
    if barrier_dominated:
        # Max allowable barrier so Z_coup(f_max) ≤ Z_tissue
        # Z_coup = 1/(ω*ε₀εᵣA/d) ≤ R_tiss  →  d ≤ 2π*f_max*ε₀*εᵣ*A*R_tiss
        omega_max_local = 2 * 3.14159 * arch.f_max_hz
        d_max_um = omega_max_local * arch.eps0 * arch.epsilon_r_barrier * arch.electrode_area_m2 * Z_tiss_approx * 1e6
        # Min area so Z_coup(f_max) ≤ Z_tissue
        A_min_cm2 = (1.0 / (omega_max_local * arch.eps0 * arch.epsilon_r_barrier * Z_tiss_approx * arch.barrier_thickness_m)) * 1e4
        if A_min_cm2 > 1000:
            area_msg = "(area-only fix impractical — increase frequency to ≥1 MHz instead)"
        else:
            area_msg = f"OR increase electrode area to ≥{A_min_cm2:.0f} cm²"
        issues.append(f"Barrier dominates signal: reduce thickness to ≤{d_max_um:.0f} µm "
                      f"{area_msg}")
    if arch.relative_noise > 0.01:
        issues.append(f"Noise {arch.relative_noise*100:.0f}% too high — "
                      f"lock-in with ≥1s integration can achieve 0.1%")
    if arch.f_max_hz < 1e5:
        issues.append(f"Bandwidth too low ({arch.f_max_hz/1e3:.0f} kHz) — "
                      f"β-dispersion peak is at 10 kHz – 1 MHz; need ≥1 MHz")
    if arch.electrode_area_m2 < 5e-4:
        issues.append(f"Electrode area {arch.electrode_area_m2*1e4:.1f} cm² too small — "
                      f"target ≥25 cm²")
    if not issues:
        issues.append("No architectural bottlenecks detected.")

    for issue in issues:
        print(f"║  • {issue:<58}║")

    print(f"╚{line}╝")
    print(f"  NOTE: Differential CRLBs above use a simplified noise model")
    print(f"  (frequency-independent, conservative). Authoritative differential")
    print(f"  CRLBs vs f_tiss are in composite_fim_differential.py.")
    print()

    return {
        "snap_isolated": snap.tolist(),
        "kalman_isolated": kf.tolist(),
        "snap_differential": snap_diff.tolist(),
        "kalman_differential": kf_diff.tolist(),
        "condition_number": float(cond_n),
        "viable_isolated": not any_fails_isolated,
        "viable_differential": not any_fails_diff,
    }


# ── Monte Carlo ────────────────────────────────────────────────────────

def monte_carlo_identifiability(arch, cole, n_samples=5000, rng_seed=42):
    rng = np.random.default_rng(rng_seed)
    nitrates  = rng.uniform(5.0,  100.0, n_samples)
    pHs       = rng.uniform(5.5,  7.5,   n_samples)
    moistures = rng.uniform(0.15, 0.35,  n_samples)
    conds  = np.zeros(n_samples)
    crlbs  = np.zeros((n_samples, 3))
    valid  = np.zeros(n_samples, dtype=bool)
    for s in range(n_samples):
        theta = np.array([nitrates[s], pHs[s], moistures[s]])
        I = fisher_information_matrix(arch, cole, theta)
        d = det(I)
        if d <= 0 or not np.isfinite(d): continue
        conds[s] = cond(I)
        try:
            crlbs[s] = np.sqrt(np.abs(np.diag(inv(I))))
            valid[s] = np.isfinite(crlbs[s]).all()
        except Exception:
            pass
    return conds[valid], crlbs[valid]


# ── Main ───────────────────────────────────────────────────────────────

def main():
    print("\nMyceSoil — Fisher Information Analysis")
    print("Sensitivity matrix is ASSERTED not measured — all CRLBs are projections.")
    print("="*64)

    cole = ColeParameters()

    # ── 1. Feasible architecture (paper defaults) ──────────────────
    print("\n[1/3] FEASIBLE ARCHITECTURE (paper Section 4.3 defaults)")
    arch_feasible = Architecture()
    r_feasible = viability_report(arch_feasible, cole)

    # ── 2. Naive architecture (thick barrier, small area, old noise) ─
    print("\n[2/3] NAIVE ARCHITECTURE (thick barrier — what fails and why)")
    arch_naive = Architecture(
        barrier_thickness_m=0.5e-3,
        electrode_area_m2=1e-4,
        f_max_hz=100_000.0,
        relative_noise=0.03,
    )
    r_naive = viability_report(arch_naive, cole)

    # ── 3. Monte Carlo over physiological range ────────────────────
    print("[3/3] MONTE CARLO identifiability (5,000 samples, feasible arch)")
    conds, crlbs = monte_carlo_identifiability(arch_feasible, cole, n_samples=5000)
    print(f"  Valid samples: {len(conds):,}/5000")
    print(f"  Condition number: median {np.median(conds):.2e}, "
          f"95th pct {np.percentile(conds,95):.2e}")
    print(f"  Nitrate CRLB:   median {np.median(crlbs[:,0]):.2f} mg/kg")
    print(f"  pH CRLB:        median {np.median(crlbs[:,1]):.3f} units")
    print(f"  Moisture CRLB:  median {np.median(crlbs[:,2]):.4f} m³/m³")

    # Save results
    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(out_dir, "fim_results.json")
    results = {
        "feasible": r_feasible,
        "naive": r_naive,
        "monte_carlo": {
            "n_valid": len(conds),
            "cond_median": float(np.median(conds)),
            "nitrate_crlb_median": float(np.median(crlbs[:,0])),
            "pH_crlb_median": float(np.median(crlbs[:,1])),
            "moisture_crlb_median": float(np.median(crlbs[:,2])),
        }
    }
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
