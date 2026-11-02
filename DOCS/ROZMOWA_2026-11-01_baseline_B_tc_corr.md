# Rozmowa 2026-11-01 — baseline dla składów A/B, poprawki V/Sc/Ti (`Tc_corr`), baza B, koniec optymalizacji GL

Kontynuacja `ROZMOWA_2026-10-29_opt_gl_results.md`. Commity użytkownika: `4e30afa` („baza B”, zawiera kod `Tc_corr`
i `databases/sra.TiNbZrHfTaMoWYVSc.kp10.ew0.6.gl.csv`), wypchnięte.

Nazewnictwo użytkownika: **skład A** = Ti, Nb, Zr, Hf, Ta, Mo, W, Y, Re, Ru; **skład B** = Ti, Nb, Zr, Hf, Ta, Mo, W, Y, V, Sc.
(Uwaga: w starszych notatkach „zbiór B” oznaczał zestaw z Re, Ru.)

## 1. Stopy ze Sc w baseline
- `results/Tc_literature.xlsx` ma 32 wiersze ze Sc: 30 z Tc_exp i wynikiem KKR GL, z czego 6 single-phase.
  Źródła: Krnel 2022 (19 wierszy), von Rohr 2018 (5), Kubo 2025 (3), Kitagawa i Seki 2024, Jangid 2024.
- Do 29.10 notebook baseline ich nie pokazywał, bo filtrował do zestawu z Re, Ru (optymalizacje `opt.ratios.gl`).
  Wcześniejsza pełna wersja (wariant A) miała osobną grupę Sc; przy uproszczeniu z 29.10 została usunięta.

## 2. `notebooks/Tc_Baseline_Comparison.ipynb` przebudowany
- Generuje go `notebooks/build_Tc_Baseline_Comparison.py`.
- **Sekcje:**
  - 0: dane (121 wierszy);
  - 1: skład A (74);
  - 2: skład B (112);
  - 3: A ∪ B (121; A ∩ B = 65);
  - 4: poprawki `Tc_corr`;
  - 5: wnioski.
- **Bez zapisu plików `.png`** (życzenie użytkownika). Stare `Tc_baseline_gl_*.png` zostały w `notebooks/`.
- **Wyniki bez poprawek (μ* = 0.1):**
  - A: τ 0.66, MAE 1.4 K;
  - B: τ 0.46, MAE 1.67 K;
  - B bez V i Sc: τ 0.67 (jak A);
  - stopy ze Sc: mediana KKR/exp 0.71, zaniżenie rośnie z udziałem Sc (6 single-phase: τ 0.97, MAE 0.56 K);
  - stopy z V: mediana 1.24, τ 0.29.
- **Dopasowanie błędu log(KKR/exp) do udziałów pierwiastków** (przy udziale 25 %): Sc ×0.53, Ti ×0.69, V ×1.28.
  Status „caution” nic nie dodaje.

## 3. Poprawki — `Tc_corr` (kod w `4e30afa`)
- **Poprawka 1 — θ_D bcc dla Ti i Sc** w regule mieszaniny: Ti 420 → 300 K, Sc 355 → 215 K.
  λ przeskalowane o (θ_mix/θ_corr)². KKR dla czystego bcc Sc daje 193 K.
- **Poprawka 2 — sprzężenie elektron–paramagnon na każdym atomie** (Bose 2009, J. Phys.: Condens. Matter 21 025602,
  równania 26–27):
  - λ_es = α Σ x_i s_i ln(1/(1−s_i)), gdzie s_i = I_i N_i(E_F) (`STONER_I_EV`, gęstość stanów KKR na atom);
  - α = 0.3;
  - Tc z równania 27, μ* = 0.1.
- **Dopasowanie:** `scripts/baseline_corrections.py` → `results/baseline_corrections.csv`, 121 wierszy A ∪ B,
  walidacja krzyżowa z pominięciem jednego autora. Zakresy w CV: θ_Ti 280–300 K, θ_Sc 185–275 K, α 0.3–0.4.
  μ* = 0.1 zostaje optymalne.
- **Wyniki (τ przed → po, w próbie / w CV):**

  | podzbiór | τ |
  |---|---|
  | A | 0.66 → 0.71 / 0.71 |
  | B | 0.46 → 0.70 / 0.63 |
  | A ∪ B | 0.47 → 0.68 / 0.62 |
  | z V | 0.29 → 0.48 / 0.42 |
  | z Sc | 0.53 → 0.74 / 0.61 |
  | z Ti ≥ 20 % | 0.31 → 0.57 / 0.48 |

  Mediana KKR/exp 1.02 → 1.10.
- **Poprawki działają tylko razem.** Samo λ_es szkodzi Sc i Ti (dopasowanie daje α = 0); samo θ_D nie usuwa zawyżenia V.
- **Ograniczenia:**
  - stałe I_i są przybliżone, a N(E_F) z KKR dla V jest za niskie;
  - czysty V w trybie mix: ok. 14 → ok. 11 K (eksperyment 5.4 K);
  - pominięcie Krnela w CV usuwa 19 z 30 stopów ze Sc.
- **Kod:**
  - `consts.py`: `TC_CORR_THETA_BCC_K`, `TC_CORR_SF_ALPHA`, `TC_CORR_MU`;
  - `process_kkr.compute_tc_corrected`: klucze `Tc_corr`, `theta_corr`, `lambda_corr`, `lambda_es`;
    θ zmienia się tylko dla `debye_mode == 'mix'`;
  - wywołania w `derive_kkr`, `_process_missing_theta` i `run_opt_compositions.recompute_debye_consistent`;
  - `database.DERIVED_KEYS` rozszerzone;
  - `Tc_mu*` bez zmian;
  - sprawdzone: zgodność z testem do 2·10⁻⁵ K, loader `--init_db` liczy `Tc_corr` dla 1774 wierszy bazy A.
- **Cel optymalizacji:** `--target Tc_corr`.

## 4. Optymalizacje GL (skład A) — dane końcowe
- Wszystkie 15 runów zakończyło się przez early stop. Poprawka OOM zadziałała: mc6, mc7 β1/β2/β10 doszły do
  48/41/44/55 iteracji. Iteracja 30 jest utracona w mc6, β0, β1 i β10.
- **Zmiany mistrzów:**
  - mc6: 13.34 → 13.52 K (Nb47Zr33 + Ta, Hf, Mo, Ti po 5 %);
  - mc7 β10: 11.63 → 11.98 K.
  - Skan β daje 11.79–11.98 K. Wnioski bez zmian.
- **`notebooks/OptGL_mu0.1_Runs.ipynb`** przeliczony. Kolumna `iters` liczy ukończone iteracje,
  a status pokazuje iteracje utracone.

## 5. Baza B (losowe składy)
- `random.ratios/sra.TiNbZrHfTaMoWYVSc.kp10.ew0.6.gl`: wszystkie 1800 składów z listy policzone bez błędów
  (brak `errorlog`, 0 nieudanych przy budowie bazy). Ustawienia jak w A: G + Lloyd, ew 0.6, bz 10/20, PBE, SRA.
- **`databases/sra.TiNbZrHfTaMoWYVSc.kp10.ew0.6.gl.csv`:**
  - 1800 × 273, układ jak w A;
  - 1770 użytecznych (30 z C' < 0);
  - kolumny `Tc_corr` usunięte z pliku, bo są przeliczane przy wczytywaniu.
- Wersji `.interior.csv` i skryptu SLURM dla B jeszcze nie ma (pytanie do użytkownika).

## 6. Top 10 składów bazy B (tryb mix, jak w optymalizatorze)

Sortowanie wg Tc bez poprawek (`Tc_mu0.1`):

| # | skład | Tc bez poprawek [K] | Tc z poprawkami [K] |
|---|---|---|---|
| 1 | V70 Ti30 | 20.08 | 18.23 |
| 2 | V57 Ti43 | 18.83 | 18.39 |
| 3 | Nb68 Zr32 | 18.70 | 17.22 |
| 4 | V88 Sc12 | 18.16 | 15.04 |
| 5 | Zr56 Nb44 | 17.50 | 15.72 |
| 6 | Zr72 Nb28 | 14.41 | 12.73 |
| 7 | V94 Nb6 | 13.85 | 10.25 |
| 8 | V95 Hf5 | 13.59 | 9.50 |
| 9 | V84 Zr16 | 13.17 | 8.97 |
| 10 | Nb79 Ti21 | 13.04 | 14.35 |

Sortowanie wg Tc z poprawkami (`Tc_corr`):

| # | skład | Tc bez poprawek [K] | Tc z poprawkami [K] |
|---|---|---|---|
| 1 | V57 Ti43 | 18.83 | 18.39 |
| 2 | V70 Ti30 | 20.08 | 18.23 |
| 3 | Nb68 Zr32 | 18.70 | 17.22 |
| 4 | Zr56 Nb44 | 17.50 | 15.72 |
| 5 | Ti58 V35 Zr6 | 13.04 | 15.10 |
| 6 | V88 Sc12 | 18.16 | 15.04 |
| 7 | Nb74 Sc26 | 11.43 | 14.73 |
| 8 | Nb79 Ti21 | 13.04 | 14.35 |
| 9 | Nb60 Sc40 | 8.58 | 13.98 |
| 10 | Nb51 Ti49 | 9.85 | 13.88 |

## 7. Uwaga użytkownika
„Wykonuj tylko te polecenia, o które proszę. O dodatkowe rzeczy się pytaj.” Przerwane zostało porównanie
binarnych stopów V z literaturą, o które użytkownik nie prosił.

## 8. Otwarte
- Czy zrobić `.interior.csv` dla B oraz skrypt SLURM z runami dla B?
- Jaki cel optymalizacji dla B: `Tc_mu0.1` czy `Tc_corr`?
