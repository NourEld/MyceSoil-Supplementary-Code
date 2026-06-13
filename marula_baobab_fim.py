#!/usr/bin/env python3
"""
MyceSoil — Marula & Baobab Xylem Parenchyma FIM Analysis
============================================================

Two African case studies with distinct wood anatomies:
  Marula (Sclerocarya birrea, Anacardiaceae): medium-hard tropical hardwood,
    ~15-25% xylem parenchyma, drought-adapted savannah species, deciduous.
  Baobab (Adansonia digitata, Malvaceae): pachycaul with ~40-60% storage
    parenchyma, extreme water-storage anatomy, up to 2000+ years lifespan.

Target analytes (African context, rainfall-driven seasonality):
  theta = [water_potential_MPa, phenological_state_AU, pathogen_index]
  water_potential: -0.3 (well-watered) to -4.0 MPa (severe savannah drought)
  phenological_state: 0.0 (dry-season dormancy/leafless) to 1.0 (wet season)
  pathogen_index: 0.0 (healthy) to 1.0 (severe infection)

CAVEAT: Cole parameters are physiologically reasoned pending Experiment 4.
"""

import numpy as np
from dataclasses import dataclass, field
import sys
import os as _os
sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))

EPS0 = 8.854e-12

@dataclass
class EmbedArch:
    electrode_area_m2: float
    barrier_thickness_m: float
    epsilon_r_barrier: float
    f_min_hz: float = 100.0
    f_max_hz: float = 1e6
    n_freqs: int = 20
    relative_noise: float = 0.001
    thermal_noise_floor_ohm: float = 5.0

    @property
    def C_coup(self):
        return self.epsilon_r_barrier * EPS0 * self.electrode_area_m2 / self.barrier_thickness_m

    @property
    def frequencies(self):
        return np.logspace(np.log10(self.f_min_hz), np.log10(self.f_max_hz), self.n_freqs)


@dataclass
class TreeCole:
    """Cole parameters for xylem parenchyma tissue."""
    R0_ohm: float
    Rinf_ohm: float
    tau_s: float
    alpha: float
    # Sensitivity matrix [R0, Rinf, tau, alpha] x [psi, pheno, pathogen]
    sensitivities: np.ndarray
    theta_base: np.ndarray


def cole_impedance(omega, R0, Rinf, tau, alpha):
    return Rinf + (R0 - Rinf) / (1.0 + (1j * omega * tau)**alpha)


def total_Z(omega, tree, arch, theta):
    dth = theta - tree.theta_base
    S = tree.sensitivities
    R0    = max(tree.R0_ohm   * (1 + S[0] @ dth), 100.)
    Rinf  = max(tree.Rinf_ohm * (1 + S[1] @ dth),  50.)
    tau   = max(tree.tau_s    * (1 + S[2] @ dth),  1e-7)
    alpha = np.clip(tree.alpha + S[3] @ dth, 0.3, 0.99)
    Z_t   = cole_impedance(omega, R0, Rinf, tau, alpha)
    Z_c   = 1.0 / (1j * omega * arch.C_coup / 2.0)
    return Z_c + Z_t


def fim(arch, tree, theta):
    omega = 2 * np.pi * arch.frequencies
    Z0 = total_Z(omega, tree, arch, theta)
    sigma = arch.relative_noise * np.abs(Z0) + arch.thermal_noise_floor_ohm
    n_p = len(theta)
    # Numerical Jacobian
    J = np.zeros((len(omega), 2, n_p))
    for i in range(n_p):
        h = 1e-5 * max(abs(theta[i]), 1e-3)
        tp = theta.copy(); tp[i] += h
        tm = theta.copy(); tm[i] -= h
        dZ = (total_Z(omega, tree, arch, tp) - total_Z(omega, tree, arch, tm)) / (2*h)
        J[:, 0, i] = np.real(dZ)
        J[:, 1, i] = np.imag(dZ)
    I = np.zeros((n_p, n_p))
    for f in range(len(omega)):
        for k in range(2):
            jfk = J[f, k, :]
            I += np.outer(jfk, jfk) / sigma[f]**2
    return I


def main():
    arch = EmbedArch(
        electrode_area_m2=1.5e-4,   # 1.5 cm² bore-constrained
        barrier_thickness_m=5e-6,   # 5 µm parylene-C
        epsilon_r_barrier=2.7,
    )
    print(f"Electrode coupling capacitance: {arch.C_coup*1e12:.1f} pF")
    print(f"  (soil case reference: 664 pF with 25cm², 100µm barrier)")

    labels = ["Water pot. (MPa)", "Phenol. state (AU)", "Pathogen idx"]

    # ---- MARULA ----
    # Sclerocarya birrea: medium-hard hardwood, deciduous, drought-adapted
    # Xylem parenchyma ~15-25% of wood volume
    # Water potential: -0.3 (wet season) to -4.0 MPa (severe drought)
    # Seasonal rhythm: rainfall-driven, lose leaves at -1.2 to -1.5 MPa
    marula = TreeCole(
        R0_ohm=9500.,    # higher than oak — denser wood, lower moisture
        Rinf_ohm=720.,
        tau_s=1.1e-4,
        alpha=0.74,
        sensitivities=np.array([
            # psi (MPa)  pheno (AU)  pathogen
            [  0.20,     -0.10,       0.04  ],  # R0/R0_base
            [  0.05,     -0.04,      -0.07  ],  # Rinf/Rinf_base
            [  0.12,     -0.05,       0.10  ],  # tau/tau_base
            [ -0.01,      0.004,      0.07  ],  # alpha (absolute)
        ]),
        theta_base=np.array([-0.5, 0.8, 0.0])  # wet season, well-watered
    )

    # ---- BAOBAB ----
    # Adansonia digitata: pachycaul, ~40-60% parenchyma (water storage)
    # Much lower wood density (~200-400 kg/m³), very high water content
    # Water potential: tight stomatal control, rarely below -1 MPa before closure
    # Very large water buffer — water potential changes SLOWLY → massive Kalman leverage
    baobab = TreeCole(
        R0_ohm=4800.,    # much lower — water-saturated spongy parenchyma
        Rinf_ohm=380.,
        tau_s=1.8e-4,    # slower relaxation — larger cells, higher capacitance
        alpha=0.68,      # broader dispersion — heterogeneous storage anatomy
        sensitivities=np.array([
            # psi (MPa)  pheno (AU)  pathogen
            [  0.28,     -0.08,       0.06  ],  # higher water-potential sensitivity
            [  0.06,     -0.03,      -0.06  ],
            [  0.15,     -0.04,       0.14  ],
            [ -0.015,     0.003,      0.10  ],
        ]),
        theta_base=np.array([-0.3, 0.7, 0.0])  # wet season baseline
    )

    trees = {"Marula": (marula, "Sclerocarya birrea"),
             "Baobab": (adaob := baobab, "Adansonia digitata")}

    # Operating points
    ops_marula = {
        "Wet season, well-watered (-0.5 MPa, pheno=0.8)": np.array([-0.5, 0.8, 0.0]),
        "Dry season onset (-1.5 MPa, pheno=0.3)":          np.array([-1.5, 0.3, 0.0]),
        "Severe drought (-3.5 MPa, pheno=0.05)":           np.array([-3.5, 0.05, 0.0]),
        "Early pathogen detection (wet, path=0.2)":         np.array([-0.5, 0.8, 0.2]),
    }
    ops_baobab = {
        "Wet season, well-watered (-0.3 MPa, pheno=0.7)": np.array([-0.3, 0.7, 0.0]),
        "Dry season onset (-0.8 MPa, pheno=0.2)":          np.array([-0.8, 0.2, 0.0]),
        "Drought stress (-1.5 MPa, pheno=0.05)":           np.array([-1.5, 0.05, 0.0]),
        "Early pathogen detection (wet, path=0.2)":         np.array([-0.3, 0.7, 0.2]),
    }

    for tree_name, (tree, spp), ops in [
        ("MARULA", (marula, "Sclerocarya birrea"), ops_marula),
        ("BAOBAB", (baobab, "Adansonia digitata"), ops_baobab),
    ]:
        print(f"\n{'='*70}")
        print(f"{tree_name} ({spp}) — snapshot CRLB")
        print(f"{'='*70}")
        for op_name, theta in ops.items():
            I = fim(arch, tree, theta)
            cond = np.linalg.cond(I)
            try:
                crlb = np.sqrt(np.abs(np.diag(np.linalg.inv(I))))
            except:
                crlb = [np.inf]*3
            print(f"\n  {op_name}")
            print(f"  Condition number: {cond:.2e}")
            for i, lbl in enumerate(labels):
                print(f"    {lbl:<22}  {crlb[i]:.4f}")

    print(f"\n{'='*70}")
    print("TEMPORAL KALMAN IMPROVEMENT — wet-season baseline")
    print("="*70)
    # African trees: rainfall-driven seasonality
    # Water potential: responds to ET demand, ~4h timescale during growing season
    # Phenological state: weeks (wet-to-dry transition), ~21 days
    # Pathogen: slow progression, ~45 days
    tau_hrs = {
        "Marula": (4.0, 21*24, 45*24),
        "Baobab": (12.0, 14*24, 60*24),
        # Baobab: water potential changes more slowly (large buffer),
        # shorter phenological timescale (more abrupt dry-season onset)
    }
    sigma_tree = np.array([0.5, 0.2, 0.1])
    dt_hours = 0.25

    for tree_name, (tree, spp), theta_nom in [
        ("Marula", (marula, "Sclerocarya birrea"), np.array([-0.5, 0.8, 0.0])),
        ("Baobab", (baobab, "Adansonia digitata"), np.array([-0.3, 0.7, 0.0])),
    ]:
        tau_psi, tau_pheno, tau_path = tau_hrs[tree_name]
        Q_base = 2 * sigma_tree**2 * dt_hours / np.array([tau_psi, tau_pheno, tau_path])
        Q_mat = np.diag(Q_base)
        I_nom = fim(arch, tree, theta_nom)
        snap = np.sqrt(np.abs(np.diag(np.linalg.inv(I_nom))))

        A = np.eye(3)
        P = np.diag(sigma_tree**2)
        for _ in range(3000):
            P_pred = A @ P @ A.T + Q_mat
            P_new = np.linalg.inv(np.linalg.inv(P_pred) + I_nom)
            if np.max(np.abs(P - P_new)) < 1e-14:
                break
            P = P_new
        kalman = np.sqrt(np.abs(np.diag(P)))

        print(f"\n  {tree_name} (τ_ψ={tau_psi}h, τ_pheno={tau_pheno/24:.0f}d, τ_path={tau_path/24:.0f}d):")
        print(f"  {'Parameter':<22}  {'Snapshot':>10}  {'Kalman':>10}  {'Improv':>8}")
        print("  " + "-"*56)
        targets = [0.15, 0.08, 0.04]
        for i, (lbl, tgt) in enumerate(zip(labels, targets)):
            flag = "✓" if kalman[i] <= tgt else "—"
            print(f"  {lbl:<22}  {snap[i]:>10.4f}  {kalman[i]:>10.4f}  "
                  f"{snap[i]/kalman[i]:>6.1f}x  target={tgt}  {flag}")


if __name__ == "__main__":
    main()
