# MyceSoil — Supplementary Code (V1 Preprint)
Preprint: Eldidy, N. (2026) 'MyceSoil: Multi-Parameter Environmental Monitoring via Fisher Information Inversion of Pleiotropic Biohybrid Impedance'. Zenodo. https://doi.org/10.5281/zenodo.20681618
Python 3.9+, numpy, scipy: `pip install numpy scipy`

Keep all files in the same directory — scripts import each other.

## Core scripts (paper results)

| Script | Section | Description |
|--------|---------|-------------|
| `fisher_information_pivoted.py` | 6.1–6.2 | Core FIM. Feasible architecture default. Viability report with ✓/⚠/✗. Note: uses simplified noise model for differential — see composite_fim_differential.py for authoritative numbers. |
| `composite_fim.py` | 6.1 | 6-parameter joint FIM. Proves non-identifiability without reference electrode (condition numbers 10²⁰–10²²). |
| `composite_fim_differential.py` | 6.1 | **Authoritative CRLBs vs f_tiss.** Proper Riccati Kalman. Paper headline numbers. f_tiss ≥ 0.15 for all targets. |
| `fuzzy_adaptive_kalman.py` | 6.3 | TSK fuzzy-adaptive Kalman under 30% sensitivity matrix drift. Calibration ratios at day 180: fixed 1.22×, fuzzy 1.12×. |
| `marula_baobab_fim.py` | 6.4 | FIM for marula and baobab. Marula: WP 0.050 MPa, pheno 0.055 AU, pathogen 0.011. |

## Analysis scripts

| Script | Description |
|--------|-------------|
| `sensitivity_robustness.py` | S matrix perturbation ±10–50%. At ±50%, 99% of 1,000 trials meet all targets. |
| `tissue_fraction_feasibility.py` | f_tiss from hyphal density. Threshold 0.15. High density + ×10 enrichment = 0.1508. |
| `error_budget.py` | Systematic error budget. Dominant terms: electrode polarisation (estimated) and soil heterogeneity (Experiment 1 target). Temperature drift negligible and cancelled by differential. |
| `fuzzy_misclassification.py` | Fuzzy robustness at transition days (75, 135). ±15 day error → ≤15% change in posterior std. |

## Key results

- **f_tiss threshold**: ≥ 0.15 (all three targets, Kalman-filtered, proper Riccati)
- **Sensitivity robustness**: 99% of trials meet targets at ±50% S error
- **Fuzzy calibration**: fixed-Q overconfident 1.22× at day 180; fuzzy 1.12×
- **Temperature**: 4th Kalman state variable — entries measured in Experiment 1

## Quick start

```bash
python composite_fim_differential.py   # paper's CRLB table
python fisher_information_pivoted.py   # viability report
python sensitivity_robustness.py       # S matrix robustness
python tissue_fraction_feasibility.py  # f_tiss feasibility
```

## Note on sensitivity matrices

All sensitivity matrix entries are **assumed from physiology**, not measured.
Every CRLB is a projection under those assumptions.
Measuring them is the primary objective of Experiment 1.
Temperature sensitivity matrix (4th state variable) pending Experiment 1.

## Contact
Nour Eldidy — nneldidy@gmail.com — Forged (Pty) Ltd, Johannesburg, South Africa
