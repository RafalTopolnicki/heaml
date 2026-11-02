# Rozmowa 2026-10-29 — baseline w wariancie GL (G + Lloyd), μ* = 0.1 vs 0.2

Kontynuacja sesji 2026-10-28 (`ROZMOWA_2026-10-28_database_opt_gl.md`). Commity `73f6062` i `d9fbe47` nadal **nie wypchnięte**.

## 1. Baseline przeliczony do GL
- Wyniki w `results/comparison.sra.ew0.6.bz10-10-20` były w starym wariancie A (cutoff `valence`, DOS site).
- Zbudowano `results/comparison.sra.ew0.6.bz10-10-20.gl` (126/126 ok, 8 s):
  `python scripts/make_slim_dataset.py results/comparison.sra.ew0.6.bz10-10-20 results/comparison.sra.ew0.6.bz10-10-20.gl`
  Te same obliczenia KKR, η/λ/Tc przeliczone w GL z fort.51 + `def=` z końcowego SCF.
- Kontrola: η_mix i λ zgodne z `scripts/eta_rmta_tests/18_recompute_baseline_gl.py` (`eta_gl.csv`) do 1e-15.
  Jedyne różnice to czyste Ru i Y, które mają `outside_range` → λ = 0 (zachowanie produkcyjne). Tych stopów nie ma
  w porównaniu literaturowym.
- λ_mix GL / A: mediana 0.69 (zakres 0.58–0.80).

## 2. Notebook `notebooks/Tc_Baseline_Comparison.ipynb`
- Pierwsza wersja (pełna: GL + sekcja 0 z wariantem A, μ* scan, Stoner, CV) liczyła się ponad 15 min. Użytkownik przerwał
  i poprosił o **prosty notebook**: baseline, eksperyment vs KKR, dopasowanie liniowe, kolejność, μ* = 0.1 i 0.2,
  **bez Stonera itp.**, z zachowaniem kolorowania scattera udziałem pierwiastka.
- Obecny notebook (po polsku, wykonanie ok. 40 s), sekcje:
  1. dane;
  2. metryki;
  3. scatter z LR, LR przez 0 i y = x;
  4. panele kolorowane udziałem pierwiastka dla μ* 0.1 i 0.2;
  5. rangi i odsetek dobrze uporządkowanych par vs próg ΔTc;
  6. tabela stopów;
  7. wnioski.

  Wykresy zapisywane do `notebooks/Tc_baseline_gl_*.png`.
- Stara pełna wersja (wariant A, μ* = 0.2, Stoner, skan μ*, CV) → `notebooks/Tc_Baseline_Comparison_A_full.ipynb`.
  Pełna wersja GL (nieużywana) jest tylko w scratchpadzie sesji.

## 3. Wyniki (zbiór B, n = 74, w tym 65 single-phase bcc; θ_D mix)

| | GL μ*=0.1 (cel optymalizacji) | GL μ*=0.2 | A μ*=0.2 (stara produkcja, ref.) |
|---|---|---|---|
| Kendall τ / Spearman ρ | 0.66 / 0.86 | 0.65 / 0.85 | 0.72 / 0.89 |
| Pearson r / R² | 0.84 / 0.71 | 0.81 / 0.65 | 0.85 |
| LR Tc_exp = a·Tc_KKR + b | 0.45x + 3.4 | 0.77x + 4.8 | — |
| LR przez 0 | 0.88 | 2.20 | 0.77 |
| MAE / bias | 1.4 K / +0.3 K | 4.3 K / −4.3 K | 1.95 K / +1.5 K |
| mediana KKR/exp | 1.05 | 0.30 | 1.22 |
| źle uporządk. pary (ΔTc_exp > 1 K) | 6.4 % | 6.3 % | 4.9 % |

Single-phase bcc: GL μ*=0.1 τ 0.69, R² 0.75, MAE 1.37 K.

- **GL przy μ* = 0.1 trafia w wartości bezwzględne**: 60/74 stopów w przedziale ×1.5.
- **Skala KKR jest za szeroka** (nachylenie LR 0.45):
  - przeszacowane: stopy Nb-bogate z wysokim λ, ×1.4–1.5 (Nb67HfZrTi33 13.3 vs 9.4 K, (TiHfTa)Nb0.9);
  - niedoszacowane: stopy z Re/Mo przy niskim λ, ×0.4–0.6 (NbReZrHfTi 2.3 vs 5.3 K, (TiHfTa)1-xNbx przy x ≈ 0);
  - stopy z Tc_exp > 8 K: mediana 1.2×.
- **GL μ* = 0.2 zaniża Tc ok. 3×** — μ* = 0.2 było dobre tylko dla wariantu A.
- **Kolejność prawie nie zależy od μ*** (τ między rankingami KKR dla 0.1 i 0.2 = 0.96; różnice wyłącznie od θ_mix).
  Wariant A porządkuje nieco lepiej (τ 0.72 vs 0.66), ale ma większy błąd bezwzględny.
- Z pierwszej, przerwanej wersji (nie ma jej w obecnym notebooku): najlepsze μ* dla GL dla wartości bezwzględnych = 0.09
  (rms log 0.28, bootstrap 0.085–0.095); τ rośnie lekko przy mniejszym μ* (0.706 @ 0.055).

## Otwarte
- `git push` commitów 73f6062 i d9fbe47, potem uruchomienie optymalizacji GL na Topoli (bez zmian od 2026-10-28).
- Notebooki i `results/` są poza gitem (`results/comparison.sra.ew0.6.bz10-10-20.gl` tylko lokalnie).
- Ewentualnie: lepsze porządkowanie przez wariant A (τ 0.72 vs 0.66) — warto sprawdzić, skąd to się bierze
  (stopy z Re/Mo przy niskim λ?).
