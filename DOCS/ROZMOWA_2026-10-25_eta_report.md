# Notatka z sesji 2026-10-25 — V, Ti, Jaśkiewicz: jedna historia, raport, Lloyd

## Pytania wyjściowe
1. V: pogorszenie dopasowania mimo „porządnego” metalu bcc.
2. Ti: nie odtwarzamy wyników.
3. Jaśkiewicz 2016: brak zgodności w podziale η na kanały sp / pd / df.
Zadania: zbiór V w wariantach A/B/G dla SRA i NRL; sprawdzić tryb `srals`; PDF krok po kroku dla jednego
pierwiastka (całka vs przesunięcia fazowe, wykresy, skala log); przejrzeć PAPERS pod kątem metody fazowej;
potem: fluktuacje spinowe zamiast Stonera, siatka k w AkaiKKR, poprawki Lloyda, plan.

## Warianty η (skrót)
- **A** = odcięcie od ostatniego węzła (produkcja, `cutoff_mode='valence'`).
- **B** = człon brzegowy Gaspariego–Gyorffy'ego przy R_MT (`surface`).
- **G** = B + skok potencjału MT przy R_MT (`gg`) = formuła fazowa GG z zerem MT, czyli to, co liczy literatura
  (GG 1972 r. 7, Butler 1976 r. 2.12, Papaconstantopoulos 1977 Tab. II, Pickett 1982 r. 4, Skriver–Mertig 1985).
- **GL** = G z całkowitą DOS Lloyda w mianowniku 1/N(E_F) zamiast sumy DOS węzłowych (nowe, dziś).

## Wyniki
### Zbiór V (18 wierszy literaturowych, θ_mix, μ*=0.2, mediana Tc_KKR/Tc_exp; τ)
| funkcje | A | B | G |
|---|---|---|---|
| SRA | 1.92 (τ 0.37) | 1.81 (0.22) | 1.18 (0.21) |
| NRL | 1.56 (0.37) | 1.70 (0.15) | 1.16 (0.20) |
Zbiór B: A 1.22, G 0.90. Offset V względem B: A 1.57 → G 1.31. λ_G/λ_A koreluje z x_Ti (−0.88), nie z x_V (−0.12).
τ w zbiorze V jest słabo określone (Tc_exp 3.7–7.2 K).

### Czysty V (bz20, SRA, a0 5.663; η w mRy/a0², sp/pd/df/suma)
- pełna całka 74.6/48.4/96.6 = 219.7; A 6.1/111.1/96.6 = 213.8 (λ 1.07, Tc 15.8 K); B 6.0/54.9/96.6 = 157.5;
  **G 4.2/41.1/103.5 = 148.7**.
- Papaconstantopoulos 1977 (RMTA): 4.1/33.8/104.0 = 141.9, λ 1.055, Tc 16.7 K; „poprawka” = df/2 → 4.6 K.
- Bose 2009 (odpowiedź liniowa FP-LMTO): η 147.6, λ_ep 1.24, Tc_McM 16.4 K; z λ_sf 0.25–0.35 → 5.0 K. Eksp. 5.4 K.
- **Wniosek: RMTA i LR dają dla V Tc ≈ 16 K; nasze 15.8 K to znany wynik, nie błąd. Brakuje fluktuacji spinowych.**
- Odcięcie A zawyża pd dla V 3×, tak jak dla Ti; to samo w sra, nrl i srals.

### srals, nrl, funkcjonał (stop Ta34Nb33Hf8Zr14Ti11, a=3.36 Å, bz20; η_mix, Jaśkiewicz 157)
| bieg | A | B | G |
|---|---|---|---|
| SRA PBE | 217 | 226 | 202 |
| SRA LDA | 238 | 226 | 201 |
| NRL PBE | 202 | 239 | 213 |
| NRL LDA | 218 | 234 | 208 |
| srals PBE | 217 | 228 | 204 |
Spin–orbita ≤1 %, relatywistyka ≤6 % (głównie sp), funkcjonał ≤2 %. `srals` działa bez zmian kodu
(`finalscf.py --rel srals`), ale podwaja liczbę punktów k (brak symetrii odwrócenia czasu) → przy bz30 przekracza limit.

### Czyste pierwiastki, zbieżność k (stare binarium)
- V: bz20 N_site 8.71/spin, N_Lloyd 10.28; bz30 9.77 / 11.41; literatura 12.7 (Papa), 12.65 (Bose).
- Nb a=6.20 nrl-LDA bz30: N_site 7.85, N_Lloyd 9.79, Papa 9.71.
- bz40 (i srals bz30) → `***err in bzmesh...too many k-points` (nk1x=10000 nieredukowalnych punktów).

### PDF: results/eta_rmta_report/report.pdf (11 stron)
Ta w stopie Jaśkiewicza, SRA vs NRL: potencjał i dV/dr (log–log), funkcje radialne (węzły, zachowanie r^γ przy
jądrze), funkcje podcałkowe (symlog), całka skumulowana M(r0) (brak plateau w SRA sp), dopasowanie fazowe, tabele
M/δ/η. Dodatki: Ti (zmiana znaku M_pd przez odcięcie: −0.30 → +0.62, η_pd 26 → 110), V, nrl/srals, Lloyd.
Diagnoza: całka objętościowa jest dobrze określona tylko dla funkcji nierelatywistycznych; dla SRA kanał sp eksploduje
(M_sp 51 zamiast 1); odcięcie A usuwa eksplozję razem z kasującymi się płatami i zmienia stosunki kanałów inaczej
dla każdego pierwiastka; G jest poprawną i parametrową wielkością.

### Lloyd — brakujący czynnik 1.3 (nowe)
- AkaiKKR liczy uogólnioną (CPA) scałkowaną DOS Lloyda: `detl` w `spckkr.f`; `spmain.f`: `total = Im detl(E_F)`,
  `def = Im d detl/dE`, drukowane w każdej iteracji SCF jako `ef= ... def= ...`. (`def` w bloku `go=dos` to
  różnica skończona na siatce rzeczywistej — bezużyteczna.)
- N_Lloyd/N_site = 1.26 ± 0.02 dla 113 stopów bazy (1.18–1.30); czyste V 1.18, Nb 1.26, Ta 1.29, Hf 1.37.
- Formuła GG ma w mianowniku całkowitą DOS na atom (APW/Lloyd), nie sumę DOS węzłowych. Po przeskalowaniu
  η_G·N_site/N_Lloyd: **stop η_mix 202 → 158.4 vs Jaśkiewicz 157.2**; Ta 162/154, Nb 158/162, Hf 181/161,
  Zr 193/176, Ti 87/126; czysty Mo 119/119, Ta 161/155.
- Dopasowanie GL: zbiór B mediana 1.16 @μ*0.10, 1.06 @0.11, 0.87 @0.13; τ 0.70; najlepsze rms log 0.266 @μ*0.105
  (A: 0.321 @0.205). λ_GL zbioru B 0.48–0.92. Zbiór V @0.11: 1.52 (A@0.2: 1.92). Zbiór B Ti≥30 % @0.11: 0.78
  (A: 1.16) → wskazuje na θ_D hcp Ti w θ_mix.

### Fluktuacje spinowe — co zamiast Stonera
- Obecny kod: S = 1/(1 − I·N) z DOS węzłową na spin i I z consts (V 0.50, Ti 0.48), μ_sf Berka–Schrieffera,
  nadwyżka ponad S_ref 1.15 dodawana do μ_base 0.185. Konwencja I niespójna, cała korekta w μ*, kalibracja na
  zawyżonym η.
- MJW (PAPERS) tabelaryzuje χ/χ0 i N(E_F) (oba spiny, /eV·atom): Sc 2.47/1.73, Ti 2.17/1.59, V 2.34/1.64,
  Y 1.88/1.41, Zr 1.64/1.28, Nb 1.72/1.40, Mo 1.23/0.65 → I = (1−1/S)/N ≈ 0.34, 0.34, 0.35, 0.33, 0.31, 0.30, 0.29 eV.
- Bose 2009 r. 26–27: λ_sf = α·N·I·ln[1/(1−N·I)], Tc = ω_ln/1.2·exp{−1.04(1+λ+λ_sf)/[λ−λ_sf−μ*(1+0.62(λ+λ_sf))]}.
  V: N·I 0.57 → λ_sf 0.49 (α=1); Bose 0.35, Rietschel–Winter 0.34; Nb 0.13–0.21, Ti 0.06, Sc/Y 0.3–0.5.
  Jeden parametr α, μ* standardowe 0.10–0.13.

## Zmiany w kodzie / plikach
- `scripts/eta_rmta_tests/`: 12 (metryki zbioru V A/B/G, tag nrl), 13 (η z fort.51 biegów nrl/srals),
  14 (figury i tabele raportu), 15 (tabela biegów testowych vs literatura), 16 (normalizacja Lloyda + fit).
- `scripts/src/consts.py`: zmienna środowiskowa `AKAIMODBIN` nadpisuje ścieżkę do specx (produkcja bez zmian).
- Kopia AkaiKKR `/home/rafal/WORK/HEA/RKKY/cpa2002v010.bigk` (zbudowana, makefile.gfortran): `bzmesh.f`
  nk1x 10000 → 400000; `spmain.f` zapisuje `detl` do `fort.52` (format `parse_fort52_detl` w macmillan.py).
- Dane: `results/test.nrl.ew0.6/V_set_nrl/*` (19 stopów nrl), `TNHZT_a3.36_srals`, `V_srals`, `V_*_bz30`,
  `Nb_a6.20_*`; `results/comparison.sra.ew0.6.bz10-10-20/eta_{valence,surface,gg}_nrl.csv`, `eta_channels_nrl.csv`,
  `v_set_abg_*`, `lloyd_vs_site_dos.csv`; `results/test.kconv/` (zbieżność k nowym binarium — w toku w chwili
  zapisu: V bz20/40/50/60, Nb bz40/60, stop bz40).
- Raport: `results/eta_rmta_report/report.tex|pdf`, `tab_*.tex`, `fig/`, `extra_runs_table.txt`.

## Plan (kolejność)
1. Produkcja: η = wariant G z mianownikiem N_Lloyd (`def=` z ostatniej linii SCF lub fort.52); przeliczyć bazę
   i optymalizacje z fort.51.gz; dopasować μ* (spodziewane ≈0.11).
2. θ_D bcc dla Ti/Zr/Hf/Sc w θ_mix; test na stopach Ti≥30 %.
3. λ_sf według Bose z I z MJW, jeden parametr α (CV na zbiorze B+V); usunąć stary Stoner z μ*.
4. Zbieżność k (test.kconv): czy N_Lloyd czystych pierwiastków zbiega do Papaconstantopoulosa przy bz50–60;
   dla stopów bz20 prawdopodobnie wystarczy.
5. Ponowna synteza Ti/V, aktualizacja VALIDATION_NOTE i README w eta_rmta_tests.

---

## Dopisek (sesja 2, 2026-10-25 wieczorem): zbieżność k dla czystego V

### Co się stało z biegami
`run_bigk.sh` (V bz20/40/50/60 → Nb a6.20 nrl bz40/60 → stop TNHZT bz40) został przerwany ok. 15:00 w bloku DOS biegu
V bz60. SCF bz60 zbiegł, `fort.51` jest kompletny, tylko `.out` ucięty. Biegi Nb i stopu nie wystartowały (brak `finished`).

### Wynik (SRA PBE, a0 5.663; N oba spiny w 1/Ry, η w mRy/a0²; tabela `results/test.kconv/kconv_table.txt`,
skrypt `scripts/eta_rmta_tests/17_kconv_table.py`)

| bzqlty | nk | N_site | N_Lloyd | η_G sp/pd/df/tot | η_GL | η_A |
|---|---|---|---|---|---|---|
| 20 (prod.) | 1661 | 17.4 | 20.5 | 4.2/41.1/103.5/148.7 | 126 | 214 |
| 30 (stare bin.) | 5216 | 19.5 | 22.8 | 5.1/47.0/116.7/168.8 | 144 | 244 |
| 40 | 11921 | 18.8 | 22.0 | 4.7/43.2/113.6/161.5 | 138 | 230 |
| 50 | 22776 | 19.4 | 22.6 | 5.1/45.7/117.3/168.1 | 144 | 241 |
| 60 | 38781 | 19.3 | 22.6 | 5.0/45.6/117.2/167.8 | 144 | 240 |
| Papa 1977 / Bose 2009 | | | 25.4 / 25.3 | 141.9 / 147.6 | | |

- Zbieżność przy bz50 (bz50 ≈ bz60 w 0.3 %); bz40 jest lokalnym dołkiem, zbieżność nie jest monotoniczna.
- bz20 (produkcja) zaniża czyste V o ~10 % (N_site 17.4 vs 19.4, η_G 149 vs 168). N_site/N_Lloyd = 0.85 niezależnie od siatki.
- Po zbieżności **η_GL = 144 między Papaconstantopoulosem (141.9) a Bose (147.6)** — wariant G z mianownikiem Lloyda
  odtwarza literaturę bez parametrów.
- N_Lloyd zbiega do 22.6, nie 25.4: pozostałe 11 % to metoda (MT/SRA/ew), nie k-punkty; w η_GL w dużej mierze się znosi.
- Stopy (hipoteza, że CPA wygładza DOS i bz20 wystarcza) i Nb a6.20 nadal nie sprawdzone.

### Siatka k w AkaiKKR (bzmesh.f, nfqlty.f)
Dla bcc `nf = bzqlty`, krok `d = 1/(2 nf)`, indeksy −nf+1…nf: zawsze siatka Γ-centrowana o 2·nf podziałach, zawierająca Γ
i punkty brzegu strefy. Parzyste i nieparzyste bzqlty to ta sama rodzina siatek (brak przesunięcia), różnią się tylko gęstością.
Nieparzyste wartości służą więc do oszacowania rozrzutu wokół granicy, nie do testu innej klasy siatki.

### Decyzje i biegi
- **Nie podnosić bzqlty powyżej 60** (decyzja użytkownika).
- Uruchomiono V bz15/25/35/45/55 (binarium bigk, `results/test.kconv`, skrypt w scratchpadzie `run_bigk_odd.sh`,
  znacznik `finished_odd`); po zakończeniu `python scripts/eta_rmta_tests/17_kconv_table.py > results/test.kconv/kconv_table.txt`.
- Do zrobienia: Nb a6.20 nrl-vbh bz40/60 i stop TNHZT bz40 (vs bz20) — test hipotezy CPA.

---

## Dopisek (sesja 2, c.d.): seria V kompletna, test siatki stopu na Topoli, decyzje o η i λ_sf

### Zbieżność k czystego V — wynik końcowy (`results/test.kconv/kconv_table.txt`)
| bzqlty | η_G | η_GL |
|---|---|---|
| 15 | 184 | 161 |
| 20 (produkcja) | 149 | 126 |
| 25, 30, 35, 45, 50, 55, 60 | 168–170 | 144–146 |
| 40 | 161.5 | 138 |
| Papa 1977 / Bose 2009 | 141.9 / 147.6 | |

Od bz25 wszystkie siatki poza bz40 leżą w paśmie ±1 %: granica η_G ≈ 168, η_GL ≈ 145, N_Lloyd 22.6 (lit. 25.4).
bz20 i bz40 odstają w dół, bz15 w górę — produkcyjne bz20 to pech jednej siatki, nie trend. Siatka AkaiKKR
(bzmesh.f): nf = bzqlty, 2·nf podziałów, Γ-centrowana, zawsze z Γ i brzegiem strefy → parzyste/nieparzyste to ta
sama rodzina, nieparzyste służą tylko do oceny rozrzutu.

### Test siatki dla stopu — na Topoli (Topola liczy, analiza lokalnie)
- Commit `315c682` (push na `stoner`): `slurm/topola_kconv.sbatch` (tablica, jeden finalscf.py na bzqlty, 14 cpu,
  8 GB, 12 h, odmawia bz > 60), `slurm/submit_kconv.sh` (parametry przez env, w tcsh `env VAR=x`),
  `slurm/akaikkr_bigk.patch` (nk1x 400000 + fort.52; nakłada się czysto na cpa2002v010.potential2026),
  `slurm/README.md` (sekcja + kroki budowy bigk + rsync), `scripts/src/consts.py` (AKAIMODBIN).
- Uruchomiono tablicę **9383241** (8 zadań, Ta34Nb33Hf8Zr14Ti11 a=3.36 Å, bz 20 25 30 35 40 45 50 60),
  wyniki w `results/test.kconv/TNHZT_a3.36_sra_pbe_bz<N>/` na Topoli.
- Po COMPLETED: `rsync -av topola:/lu/topola/home/rtop/heaml/results/test.kconv/ results/test.kconv/`
  i lokalnie `python scripts/eta_rmta_tests/17_kconv_table.py results/test.kconv/TNHZT_a3.36_sra_pbe`
  (skrypt uogólniony: czyta composition.json, liczy N_site, N_Lloyd, η_G/η_A per pierwiastek, η_mix, η_GL).
- Zasada: Topola jest wyłącznie maszyną obliczeniową; analiza na stacji roboczej.
- `scripts/eta_rmta_tests/` nadal NIE jest w gicie (do dodania).

### Doprecyzowanie: SRA jest OK, zmienia się sposób liczenia M
Zarzut użytkownika (tab. 5 raportu): SRA G ≈ NRL G, więc czemu „SRA nie jest ok”? Odpowiedź: tryb relatywistyczny
SRA zostaje. Wrażliwa na funkcje SRA (r^γ przy jądrze) jest tylko całka objętościowa (pełna: η_sp 41272; odcięta A:
inne proporcje kanałów niż z NRL). Wariant G zależy tylko od wartości i pochodnych log. na sferze MT i daje z SRA i NRL
to samo (pd, df zgodne; różnica tylko w małym sp). Rekomendacja = zamiana A → G plus mianownik Lloyda, nic więcej.

### Parametry: teraz vs propozycja
| parametr | teraz | propozycja |
|---|---|---|
| reltyp / xc / ew / mxl / edelt / pmix | sra / pbe / 0.6 / 3 / 0.001 / 0.01 | bez zmian |
| element macierzowy M | całka od ostatniego węzła (`mcmillan_cutoff_mode='valence'`, A) | formuła powierzchniowa GG + skok MT (`'gg'`, G) |
| 1/N w η | suma DOS węzłowych (fort.51) | DOS Lloyda `def=` z ostatniej iteracji SCF |
| bzqlty finalscf | 20 | 20, warunkowo 30 (zależnie od testu stopu) |
| μ* | 0.2 (`Tc_mu0.2`) | ≈ 0.11, refit na zbiorze B |
| korekta Stonera | wyłączona (`--stoner_beta 0`); kalibracja S_REF 1.15 / MU_BASE 0.185 na zawyżonym η | usunąć; ewentualnie λ_sf (Bose) później |
| θ_D | θ_mix, λ_mix = λ_KKR·(θ_KKR/θ_mix)² | bez zmian; potem θ_D bcc dla Ti/Zr/Hf/Sc |
| Tc, DG_T_K | McMillan z θ_mix; 1500 K | bez zmian |

Zamiana η (M i mianownik) + refit μ* da się przeliczyć z istniejących fort.51.gz i .out.gz, bez nowych rachunków KKR.

### λ_sf według Bose 2009 — na czym polega i DECYZJA
- Paramagnony: w kanale singletowym odpychające (osłabiają parowanie), ale zwiększają masę → wchodzą do Tc dwa razy:
  Tc = ω_ln/1.2 · exp{−1.04(1+λ+λ_sf)/[λ − λ_sf − μ*(1+0.62(λ+λ_sf))]} (Daams–Mitrović–Carbotte; Bose r. 27).
- λ_sf = α·N·I·ln[1/(1−N·I)] = α·N·I·ln S (Bose r. 26; N na spin, I Janaka, α ≤ 1 jedyny parametr).
  Bose: V λ_ep 1.24 → Tc 16 K; λ_sf 0.25–0.35 (Rietschel–Winter 0.34) daje 5 K. Nb 0.13–0.21, Ti 0.06, Sc/Y 0.3–0.5.
  Nasze V: N·I ≈ 0.57, S ≈ 2.3 → N·I·ln S ≈ 0.48 przy α=1 → α ≈ 0.7, by trafić 0.34.
- I w spójnej konwencji z MJW: Sc 0.34, Ti 0.34, V 0.35, Y 0.33, Zr 0.31, Nb 0.30, Mo 0.29 eV.
- Otwarte przy wdrożeniu: N·I dla stopu CPA (Σ c_i N_i I_i vs N_mix·I_mix), DOS węzłowa (nie Lloyda) do Stonera,
  rozbieżność ln przy N·I → 1 (Sc/Y), kolejność: najpierw μ* na zbiorze B bez V/Sc, potem α.
- **DECYZJA (użytkownik): najpierw bez λ_sf.** Zbiór B nie potrzebuje λ_sf (GL @μ*0.11: mediana 1.06, τ 0.70,
  MAE 1.3 K); λ_sf poprawia tylko V i stopy bogate w Sc/Y. Wynik „bez fluktuacji” = punkt odniesienia; V/Sc z jednym α
  jako osobny dodatek. λ_sf wraca do produkcji dopiero, gdy V lub Sc miałyby wejść do zestawu optymalizacyjnego
  (inaczej algorytm będzie je faworyzował przez zawyżone Tc, jak wcześniej Sc).

### Plan (zaktualizowany)
1. Parser `def=` z ostatniej iteracji SCF (.out.gz) + tryb `gg` → przeliczyć bazę i optymalizacje w wariancie GL;
   refit μ* na zbiorze B (oczekiwane ≈ 0.11). Bez czekania na Topolę.
2. Wynik tablicy 9383241 → decyzja bz20 vs bz30 dla finalscf; ewentualnie tani test czystych pierwiastków
   (V, Nb, Ta, Mo, W, Zr, Ti, Hf, Y, Re, Ru) η_GL @bz30 vs Papaconstantopoulos.
3. θ_D bcc dla Ti/Zr/Hf/Sc w θ_mix (stopy Ti ≥ 30 % w GL: 0.78).
4. Dodatek: λ_sf (Bose) z jednym α dla V/Sc — poza produkcją.
5. `git add scripts/eta_rmta_tests`; README tego katalogu, VALIDATION_NOTE, raport PDF (tab_kconv.tex jest pusty).

---

## Dopisek (sesja 2, c.d. 2): parser `def=`, wariant GL wdrożony w kodzie, baza przeliczona

### Co to jest `def=`
Linia `ef= … def= …` w wyjściu AkaiKKR (spmain.f, format 2500), drukowana na końcu bloku SCF (`go`), po jednej wartości
na spin, jednostka stany/Ry/komórkę/spin. `detl(E)` (spckkr.f) = (1/π)·Im[Σ_k w_k ln det M(E,k) − Σ_i c_i faza_i] =
scałkowana DOS z formuły Lloyda (CPA wg Kaprzyka–Bansila) na konturze zespolonym; `total = Im detl(E_F)` = liczba
elektronów (warunek neutralności, `neutrl`), `def = Im[(detl(E_mse) − detl(E_mse−1))/(E_mse − E_mse−1)]` = całkowita DOS
przy E_F. To jest N(E_F) z mianownika formuły GG (APW u Papaconstantopoulosa, Lloyd w Krakowie). Brać TYLKO z bloku SCF;
`def` w bloku `go=dos` (siatka rzeczywista, różnica skończona) jest bez sensu (V: 14.7 zamiast 10.3).

### Wariant GL — definicja
| wariant | M_{l,l+1} | 1/N |
|---|---|---|
| A (produkcja) | całka objętościowa od ostatniego węzła (`'valence'`) | Σ dosef węzłowe składnika |
| B | człon brzegowy GG na R_MT (`'surface'`) | Σ dosef |
| G | B + skok potencjału na sferze (`'gg'`) | Σ dosef (własne N_i składnika) |
| **GL** | jak G | **N_Lloyd (`def=`)**, ten sam dla wszystkich składników |
Tryb `gg` nie całkuje numerycznie: M z u_l, u_{l+1}, ich pochodnych (splajn w ln r), V(R⁻) i E_F na sferze MT
(`compute_M_surface` + `(0 − V(R⁻))·u_l(R)·u_{l+1}(R)`), równoważne postaci fazowej GG sin²(δ_{l+1}−δ_l) względem zera MT
(skrypt 11). Funkcje są SRA z produkcyjnego fort.51 — KKR bez zmian, zmienia się tylko obróbka.

### Kluczowe ustalenie: normalizacja per składnik
Wzór 2 Jaśkiewicza 2016: η_i = Σ_l (2l+2) n_l^i n_{l+1}^i /[(2l+1)(2l+3) N(E_F)] |M|² z **całkowitą** N(E_F) stopu dla
każdego składnika → η_i^GL = X_i / N_Lloyd. Skrypt 16 (wczoraj) mnożył wszystkie składniki jednym średnim czynnikiem
N_site_avg/N_Lloyd (GL_avg) — dla czystych pierwiastków bez różnicy, dla stopu błędne.

Ta34Nb33Hf8Zr14Ti11 (mRy/a0², bz20):
| | A | G | GL (X_i/N_Lloyd) | GL_avg (skr. 16) | Jaśkiewicz |
|---|---|---|---|---|---|
| Ta | 232 | 207 | 148 | 162 | 154 |
| Nb | 239 | 202 | 159 | 158 | 162 |
| Hf | 161 | 231 | 156 | 181 | 161 |
| Zr | 185 | 246 | 182 | 193 | 176 |
| Ti | 187 | 111 | 123 | 87 | 126 |
| η_mix | 217 | 202 | 154.5 | 158.4 | 157.2 |
Wszystkie składniki w 4 %; wczorajsze „Ti 30 % za nisko” było artefaktem GL_avg. Czyste (bz20) GL/lit: Mo 1.00,
Ti bcc 1.02, Ta 1.04, V 0.89 i Nb 0.89 (siatka niezbieżna; V @bz50 → 1.02), Zr 1.28 (vs 96 Papa).

### Fit do eksperymentu (skrypt 18, θ_mix, λ_mix = λ(θ_KKR/θ_mix)²; `fit_gl.csv`)
| podzbiór | n | wariant | μ* | mediana | τ | MAE [K] | rms log | λ zakres |
|---|---|---|---|---|---|---|---|---|
| zbiór B | 74 | A | 0.20 | 1.22 | 0.72 | 1.95 | 0.33 | 0.64–1.25 |
| zbiór B | 74 | GL | 0.11 | 0.95 | 0.66 | 1.33 | 0.35 | 0.46–0.92 |
| zbiór B | 74 | GL | 0.09 (min rms log) | 1.15 | | 1.63 | 0.29 | |
| B, Ti≥30 % | 10 | GL | 0.11 | 0.63 | 0.76 | 1.94 | | |
| z V | 18 | A / GL | 0.20 / 0.11 | 1.92 / 1.13 | 0.37 / 0.26 | 4.7 / 1.3 | | |
| Sc bez V | 29 | A / GL | 0.20 / 0.11 | 1.07 / 0.62 | | | | |
| B + V | 92 | A / GL | 0.20 / 0.11 | 1.27 / 0.98 | 0.51 / 0.58 | 2.5 / 1.3 | | |
Mediana GL zbioru B vs μ*: 0.08:1.24, 0.09:1.15, 0.10:1.05, 0.11:0.95, 0.12:0.85, 0.13:0.77 → μ* ∈ 0.09–0.11,
wybór to decyzja (proponowane 0.10). Fit używa wyłącznie λ_GL (X_i/N_Lloyd); GL_avg tylko w csv do porównania.
Offset V względem B spada z 1.57 (A) do ~1.2 (GL). Słabe: Ti ≥ 30 % (0.63) i Sc (0.62) → θ_D hcp w θ_mix, nie η.
λ_GL/λ_A w bazie: mediana 0.67 (skrajne = czyste pierwiastki bez θ_D).

### Zmiany w kodzie (NIEZACOMMITOWANE)
- `scripts/macmillan.py`: `parse_scf_lloyd_dos(path)` (plain/gz, ostatnia linia `ef=…def=` przed `go=dos`),
  `find_finalscf_out(workdir)`.
- `scripts/macmillan_cutoff.py`: `compute_one_combination_cutoff(..., total_dos=None)` (mianownik `N_denom`),
  `sweep_combinations_cutoff(..., total_dos)`, `run_mcmillan_cutoff_sweep(..., dos_normalization='site'|'lloyd')`
  (kolumny `N_denom`, `dos_normalization`); test na V bz20: η_GL 0.1261 = 0.1487·8.712/10.277 ✓.
- `scripts/process_hea.py`: przekazuje `KKR_PARAMS_FINALSCF_PARAMS['mcmillan_dos_normalization']`.
- `scripts/src/consts.py`: nowy klucz `'mcmillan_dos_normalization': 'site'` (produkcja bez zmian; GL = `'gg'` + `'lloyd'`).
- `scripts/eta_rmta_tests/18_recompute_baseline_gl.py` → `eta_gl.csv` (126 katalogów), `fit_gl.csv`.
- Skrypty: 16 = GL_avg (nieaktualny), 17 = tabela zbieżności k, 18 = GL per składnik + fit.

### DECYZJA (użytkownik): używamy GL = X_i/N_Lloyd (skrypt 18)
Do wdrożenia: (1) `consts.py`: `mcmillan_cutoff_mode='gg'`, `mcmillan_dos_normalization='lloyd'`; (2) cel μ* = 0.10
(`Tc_mu0.1` już w results.json); (3) przeliczyć optymalizacje z fort.51.gz + .out.gz; czyste V/Nb w bazie przy bz30/50;
(4) osobno: θ_D bcc Ti/Zr/Hf/Sc w θ_mix. Uwaga: tryb `'lloyd'` wymaga `finalscf_*.out[.gz]` w katalogu finalscf
(process_hea gzipuje po kroku McMillana — kolejność OK). Topola: tablica 9383241 (stop, bz 20–60) w toku.

## Sesja 3 (c.d.) — zbieżność k dla stopu (Topola 9383241, rsync do results/test.kconv)
- Stop Ta34Nb33Hf8Zr14Ti11, a=3.36 Å, bz 20/25/30/35/40/45/50/60: wszystko zbieżne już przy bz20
  (η_mix_G 204.2–204.5, η_mix_GL 156.2–156.4, η_mix_A 217.1–217.3, N_Lloyd 22.73–22.75; kanały sp/pd/df per składnik ±0.1).
  Brak efektu „pechowej siatki” jak dla czystego V (bz20 −11 %, bz40 −4 %). Składniki GL bz20: Ta 150, Nb 161, Hf 158,
  Zr 184, Ti 124 vs Jaśkiewicz 154/162/161/176/126; η_mix 156.2 vs 157.2 (baza przy a=3.365 Å: 154.5).
- Czas na węźle 28 rdzeni: bz20 10 min, bz25 19, bz30 32, bz35 50 min, bz60 ~3.6 h.
- DECYZJA (rekomendacja): stopy → bz20 bez zmian (baza, zbiory losowe, optymalizacja); czyste pierwiastki/referencje → bz50.
- Skrypt 17: kolumna GL poprawiona na właściwe GL (η_i = X_i/N_Lloyd; wcześniej GL_avg = η_G·N_site/N_Lloyd, dla stopu 160 zamiast 156).
  Tabele: results/test.kconv/kconv_table.txt (V), kconv_table_TNHZT.txt (stop). Raport: tab_kconv.tex uzupełniony, report.pdf przebudowany.
- Commit ac0f8cc: make_slim_dataset.py, --sample_offset, topola_random_array.sbatch, submit_random_gl_TiNbZrHfTaMoWYReRu.sh, README (plan 1308 składów).

## Sesja 4 — start nowych zbiorów losowych na Topoli (wieczór 25.10)
- consts.py przełączone na produkcyjny wariant GL: `mcmillan_cutoff_mode='gg'`, `mcmillan_dos_normalization='lloyd'`
  (commit b918db5, użytkownik). Kontrola przed startem: k-punkty 10/10/20 (sieć/Debye/finalscf, jak baza; stop zbieżny
  przy bz20), pipeline z nowymi ustawieniami odtwarza η_GL stopu testowego (Ta 150.0, Nb 160.9, Hf 158.1, Zr 183.9,
  Ti 124.3, η_mix 156.2), results.json ma Tc_mu0.1/0.2/0.3 → μ*=0.10 wybierane flagą `--target Tc_mu0.1`.
- Limity Topoli (`sacctmgr show qos`): 128 zgłoszonych / 64 liczonych jobów na użytkownika, ~512 rdzeni (37 zadań × 14);
  każde zadanie tablicy = osobny job. Pierwsza tablica 1–200 odrzucona (`QOSMaxSubmitJobPerUserLimit`).
- Błąd 1 (mój): `sbatch --export=ALL,ELEMENTS=Ti,Nb,...` dzieli wartość na przecinkach → ELEMENTS=Ti → każda próbka to
  czysty Ti (errorlog pełen `Ti1.0000`), ściany padały na `--n_nonzero`. Tablice 9385641–9385649 anulowane, katalog .gl usunięty.
- DECYZJA (użytkownik): składy generowane raz, lokalnie, do pliku w gicie; zadania SLURM biorą zakresy wierszy;
  brakujące = wiersze bez results.json. Implementacja (commity 86d4330, 18b959d):
  * `scripts/composition_list.py generate|status|missing` (te same generatory i ziarna co run_random_compositions;
    nazwy unikalne, suma dokładnie 1, zera zostają zerami; dla k=11 wszystkie frakcje ≥0.05 to p≈3e-4 → max_tries 1e6).
  * `slurm/compositions/TiNbZrHfTaMoWYReRu_gl.csv` — 1308 wierszy: 800 pełny sympleks (ziarna 2000..2799), k9/8/7 po 100
    (49000/48000/47000), k6..2 dopełnienia 68/45/30/25/40 (46000…42000). Pokrycie podzbiorów: k9 10/10, k8 45/45, k2 40/45.
  * `slurm/compositions/TiScNbZrHfTaMoWYReRuLa_gl.csv` — 1001 wierszy: +1 pełny (3000), k11..2 po 100 (50000+1000k);
    k11 12/12, k10 66/66, k2 66/66. Stare 999 wpisów = pełny sympleks, ziarna 100..1099.
  * `run_random_compositions.py --composition_file FILE --sample_offset A --number_of_samples N [--dry_run]`.
  * `slurm/topola_complist_array.sbatch` + `slurm/submit_complist.sh` + `submit_random_gl_<pierwiastki>.sh`
    (NTASKS 55 / 42 zadań × 24 wiersze, 4 stopy naraz, limit 2 doby; `env FILE=<missing.csv> NTASKS=5 ...` do dopełnień).
    Skrypty oparte na ziarnach (topola_random_array.sbatch, submit_random_families.sh) usunięte.
- URUCHOMIONE: tablica **9386076** (rnd10, 55 zadań): 37 liczy się, 18 czeka na rdzenie. Szacunek: runda 4 stopów
  10-składnikowych 4–6 h, zadanie 24–36 h, cały zbiór ~2–2.5 doby (dwie fale). Zbiór 12-el jeszcze nie zgłoszony
  (42 zadania; 55+42 < 128).
- Po obliczeniach: rsync `.gl`, `composition_list.py status`, `make_slim_dataset.py <set>.gl <set>.slim.gl`
  (łączy ze starymi wpisami), optymalizacja `--target Tc_mu0.1 --initdir <set>.slim.gl --debye_mode mix`.
- Monitoring tablicy 9386076 (37 zadań w toku): 134 → 148 katalogów (37 × 4) po starcie; `sstat` wymaga surowego
  numeru zadania (`squeue -j 9386076 -r -t R -o "%A %K %M %N"`, np. 9386077.batch: AveCPU 35:45, MaxRSS 618 MB/proces);
  CPULoad węzła t11-9 (2 zadania, 28 rdzeni): 5.5 w fazie rozruchu, **18.8 po kilkunastu minutach** = 9.4 rdzenia na
  zadanie, 2.3 na proces specx — zgodne z testem 23.10. DECYZJA: CONCURRENT=4 zostaje, zbiór 12-el zgłaszać z
  domyślnymi ustawieniami. Następny punkt kontrolny: pierwsze results.json po 3–5 h, `composition_list.py status`.
