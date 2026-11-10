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

## 4. Uruchomienie
- 2026-11-03: wysłane na Topoli, joby 9395429–9395446 (6 runów × 3 w łańcuchu); w chwili sprawdzenia wszystkie PD (Priority/Dependency).
- Jest 18 jobów: 6 runów (mincomp 4, 5, 6, 8, 9, 10) × 3 w łańcuchu `afterany`. Pierwszy job każdego runu czeka na priorytet
  (Priority), a dwa kolejne na poprzedni job (Dependency). Wyniki trafią do
  `opt.ratios.gl/sra.TiNbZrHfTaMoWYVSc.kp10.ew0.6.gl.corr2.mu0.1.acqbeta2.stonerbeta0.mixdebye.mincomp<m>.r0.05.max0.7.face0.1`.

## 5. Commit
- `671fcb6`: skrypt mc4–10, README, notebook z builderem, ta notatka. Wypchnięty na `origin/main`.

## 6. Przypomnienie: czym corr2 różni się od corr
Obie wersje to McMillan z μ* = 0.1 i dwiema poprawkami:
- **θ_D bcc w θ_mix:** wybrane pierwiastki mają w regule mieszania θ_D fazy bcc; λ przeskalowane jak λ·(θ_mix/θ_corr)²;
- **λ_es:** sprzężenie elektron–paramagnon (Bose 2009), λ_es = Σ x_i α_i S_i ln(1/(1−S_i)), S_i = I_i N_i(E_F).

| | corr (2026-11-01) | corr2 (2026-11-02) |
|---|---|---|
| θ_D bcc | Ti 300 K, Sc 215 K | tylko Sc 215 K (Ti: hcp 420 K, jak w A) |
| α | 0.3 dla wszystkich pierwiastków | 0.4 tylko dla V |
| dopasowanie | A∪B bez binarnych z V | A∪B + 17 binarnych z V (GL), LOAO, warunek Nb–Zr > Nb–Ti > Ti–V |
| bez V i Sc | ≠ Tc_mu0.1 | = Tc_mu0.1 (skład A bez zmian) |

- **Problem z corr:** θ_bcc dla Ti obniżało θ_corr, więc rosło λ. Dla V68Ti32 θ spadało z 393 do 354 K, a λ rosło
  z 0.93 do ok. 1.15. To znosiło λ_es, więc Ti–V zostawało powyżej Nb–Zr. Do tego wersja ta była niespójna ze składem A.
- **W corr2:** λ_es na V–Ti jest podobne (ok. 0.08), ale bez zawyżonego λ Tc spada. V68Ti32 ma 20.1 K bez poprawki,
  18.4 K w corr i 13.2 K w corr2.
- Maksima na liniach binarnych: Nb–Zr 19.0 K > Nb–Ti 13.9 K ≈ Ti–V 13.8 K.
- Ti–V jest nadal ok. 2× powyżej eksperymentu (6–7 K): corr2 poprawia kolejność, nie wartość bezwzględną.
  Zgodność z eksperymentem wymagałaby λ_es ≈ 0.25.

## 7. Otwarte
- Po zakończeniu runów mc4–10 dodać je do `OptGL_B_corr_Runs.ipynb` (na razie notebook ma tylko mc 2/3/7).
