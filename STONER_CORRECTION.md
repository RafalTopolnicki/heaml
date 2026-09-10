# Stoner Spin-Fluctuation Correction to Tc

## 1. Observation: Systematic Tc Overestimation for Ti/Sc-Rich Alloys

The active-learning optimization converged on Ti/Sc-rich compositions as the most
promising superconductors. The KKR-McMillan pipeline predicted Tc ≈ 28–30 K for
the champion alloy Ti₀.₅₉₈ Sc₀.₃₂₀ Ru₀.₀₈₁ (and similar compositions from other
optimization runs). When these were synthesized and measured, Tc was found to be
below 5 K — a 5–6× overestimation.

This overestimation is far worse than for literature benchmark alloys (Nb/Ta/Hf/Zr-rich
HEAs), where the same pipeline overestimates by only 1.5–2.2×. The pattern is clear:
the more Ti/Sc-rich the composition, the worse the prediction.

## 2. Why Ti/Sc Composition Drives λ Up

The McMillan-Hopfield formula for the electron-phonon coupling parameter is:

```
λ = η_mix / (M_mix · θ_D²) · C
```

where η_mix = Σᵢ cᵢ ηᵢ is the composition-weighted electronic coupling parameter
(computed from KKR wavefunctions), M_mix is the average atomic mass, and θ_D is
the Debye temperature. C is a unit-conversion constant derived from the Debye model.

Ti/Sc-rich compositions drive λ high through two independent mechanisms, neither
of which is a calculation error:

**Mechanism 1 — Light atomic mass.**
Ti (47.9 amu) and Sc (45.0 amu) are among the lightest transition metals. The
literature alloys that KKR predicts well are Nb/Ta/Hf/Zr-rich with M_mix ≈ 73–106 amu.
For the Ti/Sc champion, M_mix ≈ 51 amu. Since λ ∝ 1/M_mix, this alone gives a 1.37×
enhancement in λ relative to a mid-weight alloy.

**Mechanism 2 — Soft BCC lattice, low Debye temperature.**
Both Ti and Sc are mechanically unstable in their BCC form at ambient conditions
(they exist naturally as HCP). When alloyed together without strong BCC-stabilizing
elements (Mo, Nb, W), the BCC phase is only marginally stable. The tetragonal shear
modulus C' = (C₁₁ − C₁₂)/2 collapses to ≈ 5.6 GPa for the champion composition —
roughly 5–10× lower than in literature alloys. This soft C' directly depresses the
Debye temperature: θ_D ≈ 220 K vs ≈ 250–300 K for literature alloys. Since λ ∝ 1/θ_D²,
this contributes a further 1.63× enhancement.

Combined, these two physical effects produce λ ≈ 3.5 for Ti/Sc-rich alloys vs λ ≈ 1.0–1.5
for literature alloys — roughly 2.3× higher, from pure physics, not from any numerical error.
A large λ feeds exponentially into the McMillan formula, pushing predicted Tc toward 30 K.

## 3. Debye Temperature Investigation: More δ Values Do Not Help

An early hypothesis was that the Debye temperature was underestimated due to
a numerical issue in how the elastic constants were computed. The production pipeline
used a single one-sided strain δ = 0.005 to compute the tetragonal shear modulus C'.
Better practice uses two-sided strains (which cancel the linear A₁ contamination) and
multiple δ values (to assess anharmonicity and fit a quartic polynomial).

A dedicated rerun was performed for the champion composition using:
- **Two-sided strains** (±δ), eliminating linear-term bias
- **Four δ values**: 0.003, 0.005, 0.007, 0.010
- **Quartic polynomial fit** for C'

Results from the improved Debye calculation:

| | Old (one-sided, δ=0.005, linear) | New (two-sided, multi-δ, quartic) |
|---|---|---|
| θ_D | 225.0 K | 219.1 K |
| λ | 3.41 | 3.60 |
| Tc (μ*=0.2) | 29.95 K | 29.91 K |

The fit data revealed why the improvement was negligible: C₄₄ (monoclinic) is
perfectly flat with strain — the original single-point estimate was already reliable for
that mode. C' (tetragonal) is anharmonic, and the quartic fit gives C'_raw = 5.58 GPa
vs 6.44 GPa from the old method (~13% lower). However, the lower C' is partially
offset by the two-sided correction, and the net change in θ_D is only −3%.

**Conclusion: θ_D ≈ 220 K is physically real for BCC Ti/Sc. The soft BCC lattice
is not an artifact of the Debye numerics — it reflects genuine near-instability.**
Adding more strain points improves numerical hygiene but adds substantial computational
cost (4× the KKR runs per Debye calculation) with no meaningful change to Tc.

Note: the production pipeline still uses the original one-sided single-δ approach
(a TODO exists to fix the hardcoded parameters in `process_hea.py` lines 100 and 106).
For the specific case of BCC Ti/Sc alloys, fixing this would not resolve the overestimation.

## 4. Stoner Spin-Fluctuation Correction

### What the Stoner criterion is

In a metal with high density of states at the Fermi level N(EF) and a large
exchange-correlation integral I (the Stoner parameter), the paramagnetic state becomes
unstable toward ferromagnetism when:

```
I · N(EF) ≥ 1   (Stoner criterion)
```

Even below this threshold, the Stoner enhancement factor S = 1/(1 − I·N(EF)) > 1
indicates an amplified spin susceptibility — the metal is "nearly ferromagnetic."
Ti (I ≈ 0.48 eV) and Sc (I ≈ 0.46 eV) both have high Stoner parameters, and their
d-band DOS at EF is significant. The KKR-CPA calculation for the champion alloy gives
I·N(EF) ≈ 0.41, placing it in the moderately-to-strongly enhanced regime (S ≈ 1.7).

### Why spin fluctuations suppress Tc

In a nearly-ferromagnetic metal, electrons can exchange virtual spin-density waves
(paramagnons) with each other. Unlike phonon exchange (which is attractive in the
singlet Cooper-pair channel), paramagnon exchange is repulsive — it competes with
phonon-mediated pairing and suppresses Tc. This was established by Berk and Schrieffer
(1966) and is well-documented for nearly ferromagnetic metals like Pd and Ni.

The effect is captured by replacing the fixed Coulomb pseudopotential μ* with an
effective value that includes the spin-fluctuation contribution:

```
μ*_eff = μ*_Coulomb + μ*_sf

μ*_sf = λ_sf / (1 + λ_sf · γ)

λ_sf  = I·N(EF) / (1 − I·N(EF))  =  S − 1

γ     = ln(E_sf / ω_D)
```

Here E_sf is the characteristic paramagnon energy scale (the "speed" of spin fluctuations)
and ω_D = k_B θ_D is the Debye energy. The logarithm γ = ln(E_sf/ω_D) plays the same
role as ln(EF/ω_D) does in the original Morel-Anderson pseudopotential derivation: it
counts the energy decades over which the paramagnon interaction is renormalized down
to the superconducting gap scale. Because paramagnons live at E_sf < EF, the
renormalization is smaller than for the Coulomb term, meaning μ*_sf is a significant
fraction of λ_sf — not diluted away by a large logarithm.

### Why the standard μ* = 0.13 fails for Ti/Sc

The conventional choice μ* = 0.13 is calibrated against classic superconductors and
alloys where the Stoner enhancement is weak (S ≈ 1.1–1.4). For those systems, the
spin-fluctuation pair-breaking is small and can be absorbed into an empirical μ*.
Ta/Nb/Hf-rich HEAs fall in this category: KKR predicts their Tc within 1.5–2.2×.

For Ti/Sc-rich alloys, the Stoner enhancement S ≈ 1.7 generates λ_sf ≈ 0.7, giving
μ*_sf ≈ 0.38. This nearly triples the effective pair-breaking pseudopotential
(μ*_eff ≈ 0.52 vs μ*_Coulomb = 0.13) and reduces Tc_sf from 30 K to ≈ 12 K. The
remaining discrepancy relative to experiment (<5 K) likely arises from additional
phonon softening near the BCC→HCP instability (imaginary zone-boundary modes not
captured by the Debye model) or from E_sf being even lower than estimated.

### What problem the correction solves in the optimization

Without the correction, the Bayesian optimization converges on Ti/Sc-rich compositions
because the KKR oracle assigns them the highest Tc. The optimizer is rational given
the information it receives, but the oracle is systematically wrong in this region.

The Stoner correction (`Tc_sf`) provides a physically-motivated, composition-dependent
adjustment that naturally penalizes high-Stoner compositions (Ti/Sc-rich) while leaving
low-Stoner compositions (W/Ta/Nb-rich) largely unaffected. It requires no additional
KKR calculations — the necessary N_i(EF) values are already present in the
`mcmillan_cutoff_results.csv` output — and no surrogate model fitting: the
composition-dependence comes entirely from the physics formula via tabulated I_i
(Janak 1977) and element-specific paramagnon energies E_sf_i.

## 5. Calibration of the Stoner Correction (2026-09-10)

### Why the raw Berk-Schrieffer formula overcorrects stable alloys

Applying the full formula with μ*_Coulomb = 0.13 to the 36-alloy baseline gives
median Tc_sf/Tc_exp = 0.59 — the correction overshoots in the other direction.
The reason: the conventional μ* = 0.13 is the bare Coulomb pseudopotential,
but empirical fits to stable HEA superconductors prefer μ* ≈ 0.20. That gap
(0.07 in μ*) corresponds to the "typical" spin-fluctuation pair-breaking already
implicit in how μ* is calibrated against experiments. Adding the full Berk-Schrieffer
correction on top of μ*_Coulomb = 0.13 double-counts this baseline.

### Excess-correction method

Only the EXCESS spin-fluctuation pair-breaking beyond a reference Stoner level S_REF
is added to a calibrated base μ* (STONER_MU_BASE):

```
lambda_sf = S - 1
gamma     = ln(E_sf_mix / omega_D)             [composition-weighted, Option B]

mu_sf_full = lambda_sf / (1 + lambda_sf * gamma)       [full Berk-Schrieffer]
mu_sf_ref  = (S_REF-1) / (1 + (S_REF-1) * gamma)      [reference at same gamma]
mu_sf_excess = max(0,  mu_sf_full - mu_sf_ref)

mu_eff = STONER_MU_BASE + mu_sf_excess
Tc_sf  = McMillan(theta_D, lambda, mu_eff)
```

For alloys with S ≈ S_REF: mu_sf_excess ≈ 0, mu_eff = STONER_MU_BASE.
For Ti/Sc-rich alloys with S >> S_REF: positive excess grows with S.

### Calibration procedure and results

Tested against 36 literature baseline alloys from `results/Tc_baseline.xlsx`
with both Tc_exp and KKR-computed N(EF) available. Three methods compared:

| Method | μ* formula | median Tc_sf/Tc_exp | rms(log) |
|---|---|---|---|
| No correction | μ*=0.20 fixed | 1.60 | — |
| Full Berk-Schrieffer | μ_eff = 0.13 + μ_sf_full | 0.59 | 1.12 |
| **Excess (calibrated)** | **μ_eff = 0.185 + max(0, μ_sf_full − μ_sf_ref)** | **0.997** | **0.51** |

Calibrated values (in `consts.py`):

```python
STONER_S_REF   = 1.15   # excess is zero for S <= 1.15
STONER_MU_BASE = 0.185  # absorbs Coulomb (0.13) + baseline sf pair-breaking
```

The calibrated excess correction gives median = 1.00 and 18/36 alloys above 1 —
symmetric errors — with the lowest rms(log) of all tested approaches.

### Remaining limitation for extreme Ti/Sc compositions

Even with calibration, Ti/Sc-rich champion compositions (S ≈ 1.7) are still
predicted at Tc_sf ≈ 17–20 K, above the best stable alloys (8–12 K). The optimizer
will therefore still prefer Ti/Sc-rich compositions if run without additional
constraints. Additional suppression of these candidates requires the acquisition
penalty (Option C, STONER_BETA > 0).

## 6. Implementation Status

### Files

| File | What changed |
|---|---|
| `scripts/src/consts.py` | `STONER_I_EV`, `E_SF_MEV`, `STONER_N_EF_APPROX`, `STONER_S_REF`, `STONER_MU_BASE`; `TARGET='Tc_sf'` |
| `scripts/src/process_kkr.py` | `compute_stoner_correction()`: excess-correction formula; all Stoner fields in `results.json` |
| `scripts/run_opt_compositions.py` | `--stoner_beta` / `--stoner_s_threshold` CLI args; `stoner_s_mix` logged per candidate; `selection_info.json` includes all Stoner fields |

### Stoner fields written to `results.json` per composition

`Stoner_IN_mix`, `Stoner_S`, `Stoner_lambda_sf`, `Stoner_gamma`, `Stoner_E_sf_meV`,
`Stoner_E_sf_eff_meV`, `Stoner_mu_sf`, `Stoner_mu_sf_ref`, `Stoner_mu_sf_excess`,
`Stoner_mu_eff`, `Tc_sf`.

### Optimization target

`TARGET = 'Tc_sf'` is active as of 2026-09-10. Previous runs used `'Tc_mu0.2'`.

### Stoner acquisition penalty (Option C)

`stoner_penalty_factor()` in `run_opt_compositions.py` computes approximate S_mix
from tabulated `STONER_N_EF_APPROX` (no KKR) and applies:

```
penalty = exp(-stoner_beta * max(S_mix - stoner_s_threshold, 0))
```

Disabled by default (`--stoner_beta 0.0`). Enable per run with CLI:

```bash
python scripts/run_opt_compositions.py \
  ... \
  --stoner_beta 2.0 \
  --stoner_s_threshold 1.5
```

With `--stoner_beta 2.0`: a candidate at S_mix=1.7 gets penalty = exp(−0.4) ≈ 0.67;
a stable candidate at S_mix=1.3 gets penalty = 1.0 (unaffected).
The S_mix estimate and penalty are logged in `selection_info.json` and
`top_candidates.csv` for every iteration.
