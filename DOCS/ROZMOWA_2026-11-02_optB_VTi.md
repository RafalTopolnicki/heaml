# Rozmowa 2026-11-02 — `main` = `stoner`, optymalizacje składu B (Tc_mu0.1 vs Tc_corr), problem V–Ti

Kontynuacja `ROZMOWA_2026-11-01_baseline_B_tc_corr.md`.

## 1. Git
- Katalog `random.ratios/sra.TiNbZrHfTaMoWYVSc.kp10.ew0.6.gl` na Topoli można usunąć: optymalizacje czytają tylko
  `databases/...VSc...gl.csv`, a pełna kopia jest lokalnie (5460 plików, 1 670 891 521 B).
- Testowy merge `stoner` → `main` dawał konflikt w `scripts/debye.py`. Commit `b444564 debye params` z `main`
  był nieaktualny, bo `stoner` ma przepisany `debye.py`.
- Na prośbę użytkownika: `git push --force-with-lease origin stoner:main`. Teraz `main` = `stoner` = `4981034`.
- Użytkownik przełączył się lokalnie na `main`. **Od teraz pracujemy na `main`.**

## 2. Optymalizacje składu B — notebook `notebooks/OptGL_B_corr_Runs.ipynb`
- Builder: `notebooks/build_OptGL_B_corr_Runs.py`. Notebook nie zapisuje plików `.png`.
- Sześć runów w `opt.ratios.gl/`: mincomp 2, 3, 7 × cel `Tc_mu0.1` (plain) / `Tc_corr` (corr).
- Każdy punkt ma w `results.json` obie wartości, więc każdego mistrza oceniamy obiema metrykami.
- Wszystkie runy skończyły się przez early stop: plain po 16, 25 i 37 iteracjach, corr po 17, 20 i 24.
  Przeszukiwanie jest w 79–100 % lokalne.

| mincomp | mistrz bez poprawki | Tc_mu0.1 / Tc_corr [K] | mistrz z poprawką | Tc_mu0.1 / Tc_corr [K] | odległość |
|---|---|---|---|---|---|
| 2 | V68 Ti32 | 20.14 / 18.42 | Ti56 V44 | 17.53 / 18.55 | 0.34 |
| 3 | Nb57 Zr38 Ta5 | 17.90 / 16.34 | Nb61 Zr33 Ti5 | 17.55 / 16.53 | 0.10 |
| 7 | Nb62 Zr12 Ti6 Hf5 Ta5 Y5 V5 | 12.49 / 11.34 | Nb67 Zr7 Hf6 Y5 Sc5 Ti5 Ta5 | 12.00 / 12.28 | 0.10 |

- **mc2:** oba warianty kończą na V–Ti. `Tc_corr` jest prawie płaskie wzdłuż V–Ti (18.5 ± 0.05 K dla Ti34–Ti58).
  Oba runy poprawiają też krawędź Nb–Zr (plain 16 nowych punktów, corr 10) i dochodzą do Nb61 Zr39
  (18.99 / 17.36 K, jak w składzie A), ale V–Ti wychodzi wyżej.
  - Pierwsza wersja wniosku twierdziła błędnie, że do Nb–Zr nie dochodzą. Poprawione.
- **mc3 i mc7:** ten sam obszar bogaty w Nb (Nb–Zr). Wymiana mistrzów między metrykami zmienia Tc o ≤ 1.15 K.
  Przy mc7 różni się tylko dodatek 5 %: V w plain, Sc w corr.
- **Wniosek:** przy wielu składnikach poprawka nie zmienia kierunku optymalizacji. Ma znaczenie tylko w granicy
  dwuskładnikowej, czyli dla V–Ti.
- Notebook i builder nie są zacommitowane.

## 3. Nb–Ti w składzie B
- W całej bazie B są tylko 3 stopy Nb–Ti, wszystkie losowe:

  | stop | Tc_mu0.1 [K] | Tc_corr [K] |
  |---|---|---|
  | Nb79 Ti21 | 13.04 | 14.35 |
  | Nb51 Ti49 | 9.85 | 13.88 |
  | Nb34 Ti66 | 8.12 | 13.57 |

  - Nb79 Ti21 leży poza ograniczeniem max 0.7, więc nie ma go w danych optymalizatora.
- Na każdą parę pierwiastków przypadają średnio ok. 2 punkty dwuskładnikowe (100 punktów na 45 par).
- W składzie A (mc2) też był tylko 1 stop Nb–Ti (Ti62 Nb38, 8.47 K) i 0 nowych. Optymalizacje szły do Nb–Zr,
  a nie do Nb–Ti.
- Zaproponowany, niezrobiony: osobny skan krawędzi Nb–Ti.

## 4. Problem V–Ti: literatura i diagnoza
- Spis literatury z uzasadnieniem: `DOCS/LITERATURE.md` (sekcje 1–2).
- **Literatura:**
  - Matin i in. (EPJ B 2014): dla Ti–V McMillan daje ok. 20 K, eksperyment < 10 K; przyczyną są lokalne
    fluktuacje spinowe;
  - μSR na Ti55V45 potwierdza fluktuacje;
  - Rietschel i Winter (PRL 1979): w V i Nb λ_sf ≈ 0.2, Tc wyraźnie obniżone;
  - arXiv 2406.15174: efektywne μ* wynosi 0.42 dla V i 0.28 dla Nb.
- **Diagnoza na danych z runów B** (μ* = 0.1, równanie 27 Bosego):

  | stop | Tc exp [K] | Tc_mu0.1 [K] | Tc_corr [K] | λ_es teraz | λ_es potrzebne |
  |---|---|---|---|---|---|
  | Ti30 V70 | 7.16 (Ti25V75) | 20.1 | 18.4 | 0.075 | 0.25 |
  | Ti50 V50 | 7.30 | 17.8 | 18.3 | 0.083 | 0.26 |
  | Ti70 V30 | 6.14 | 13.4 | 17.2 | 0.081 | 0.27 |

  - Zawyżenie V–Ti ×2.5–2.8, wobec Nb–Zr ×1.6–1.7 i Nb–Ti ×1.4; stąd odwrócona kolejność.
  - Potrzebne λ_es jest ok. 3.3 razy większe niż obecne, prawie stałe w całym zakresie V–Ti, i zgodne z λ_sf ≈ 0.2–0.3
    z literatury.
  - Obecne α = 0.3 dopasowano bez stopów bogatych w V; s_i = I_i N_i(E_F) jest za małe.
  - Poprawka θ_D_bcc(Ti) pogarsza stopy bogate w Ti (Ti70V30: 13.4 → 17.2 K), bo w trybie mix niższe θ daje wyższe λ.
- Doświadczalne θ_D (Cheng 1962) wynosi 244–283 K, a nasze θ_mix 393–408 K.

## 5. Proponowany plan (czeka na decyzję użytkownika)
1. Policzyć w GL 17 stopów dwuskładnikowych z V (Ti–V, V–Nb, V–Ta, V–Mo) oraz czysty V, ze składów w
   `Tc_literature_V_additions.csv`, plus Nb–Ti i Nb–Zr jako odniesienie. Skrypt SLURM na wzór
   `0_compare_with_known_V.sh`.
2. Dopasować na nowo λ_es z tymi stopami w zbiorze kalibracyjnym:
   - (a) osobne α_i (albo efektywne S_i) dla V i Ti;
   - (b) jedno α z ograniczeniem na τ dla A i B;
   - walidacja krzyżowa z pominięciem jednego autora;
   - warunek akceptacji: Nb–Zr > Nb–Ti > Ti–V przy zachowanym τ dla HEA.
3. Powtórzyć optymalizacje B z nowym `Tc_corr`. Nowa baza nie jest potrzebna, wystarczy przeliczenie przy wczytywaniu.

## 6. Spis literatury i katalog DOCS/
- `DOCS/LITERATURE.md`, sekcja 3: spis wszystkich prac z eksperymentalnym Tc (dane bibliograficzne, stop, Tc).
  - Tabele wygenerowano z `Tc_literature.xlsx` (48 prac, 124 wiersze), `Tc_literature_V_additions.csv` (18 wpisów spoza
    arkusza), `papers_data/bcc_hea_superconducting_tc_summary.xls` (12 rekordów spoza arkusza) i `Tc_baseline.xlsx`
    (czyste pierwiastki).
  - Odniesienia dla stopów dwuskładnikowych z notatek (Hulm & Blaugher 1961 i in.) są niezweryfikowane.
  - Kilka starych wierszy ma w kolumnie Link pracę teoretyczną zamiast eksperymentalnej (np. Ta34Nb33Hf8Zr14Ti11 →
    Koželj 2014, a nie Jaśkiewicz 2016).
- Wszystkie notatki `ROZMOWA_*.md` przeniesione do `DOCS/`; `ROZMOWA_2026-10-25.md` z `results/eta_rmta_report/`
  dostała nazwę `ROZMOWA_2026-10-25_eta_report.md`.
- Katalog `DOSCs/` (inny, z `large_lambda_VEC_unstableBCC.md`) pozostał bez zmian.
