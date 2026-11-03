# Rozmowa 2026-11-03 — wyniki corr2 (skład B) i skrypt dla pozostałych mincomp

Kontynuacja `ROZMOWA_2026-11-02_optB_VTi.md`.

## 1. Runy corr2 (mincomp 2, 3, 7) — zakończone przez early stop
| mincomp | iteracje | mistrz | Tc_mu0.1 | Tc_corr1 (11-01) | Tc_corr2 [K] |
|---|---|---|---|---|---|
| 2 | 16 | Nb61 Zr39 | 18.97 | 17.35 | 18.97 |
| 3 | 23 | Nb54 Zr40 Ta5 | 17.88 | 16.27 | 17.88 |
| 7 | 37 | Nb67 Zr7 Ti6 Ta5 Y5 Hf5 Sc5 | 12.02 | 12.35 | 12.87 |

- mc2: V–Ti przestaje wygrywać; w Tc_corr2 V68Ti32 ma 13.20 K, Ti56V44 11.29 K (≈ 6 K poniżej Nb–Zr).
- mc3: mistrz praktycznie jak w plain (odległość 0.04). Stopy z V tracą ≈ 1.25 K.
- mc7: mistrz praktycznie jak w corr (0.01); dodatek Sc zamiast V.

## 2. Notebook `notebooks/OptGL_B_corr_Runs.ipynb`
- Builder `build_OptGL_B_corr_Runs.py`: trzy warianty plain / corr / corr2, metryki `Tc_mu0.1`, `Tc_corr1`, `Tc_corr2`.
- Obie wersje Tc_corr przeliczane dla każdego punktu przez `process_kkr.compute_tc_corrected` z podmienionymi
  parametrami (zgodność z zapisanymi wartościami 1e-14, assert w notebooku). Wnioski (§9) przepisane. Bez plików png.

## 3. Skrypt startowy dla pozostałych mincomp
- `slurm/submit_opt_gl_TiNbZrHfTaMoWYVSc_corr2_mc4-10.sh`: mincomp 4, 5, 6, 8, 9, 10, cel Tc_corr (corr2), β 2,
  CHAIN 3, 6 × 28 cpu. Wpis w `slurm/README.md`. Sprawdzony `DRY=1`. Nie zacommitowany, nie uruchomiony.
