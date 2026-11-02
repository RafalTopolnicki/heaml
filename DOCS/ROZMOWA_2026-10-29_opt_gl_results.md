# Rozmowa 2026-10-29 (wieczór) — wyniki optymalizacji GL, porównanie ze starą optymalizacją, OOM na Topoli

Kontynuacja `ROZMOWA_2026-10-29_baseline_gl.md`. Dane: `opt.ratios.gl/` (15 runów z `slurm/submit_opt_gl_TiNbZrHfTaMoWYReRu.sh`,
skopiowane z Topoli ok. 29.10 02:10) oraz `slurm/logs/opt_*.out`.

## 1. Nowy notebook `notebooks/OptGL_mu0.1_Runs.ipynb`
- Builder: `notebooks/build_OptGL_mu0.1_Runs.py`. Przebudowa:
  `python3 build_OptGL_mu0.1_Runs.py OptGL_mu0.1_Runs.ipynb && jupyter nbconvert --to notebook --execute --inplace OptGL_mu0.1_Runs.ipynb`
  (ok. 40 s).
- Czyta `results.json` i `selection_info.json` bezpośrednio z `iteration_*/computation.tar`; nic nie trzeba rozpakowywać
  (katalogi `computation/` są puste).
- Sekcje:
  - 1–9: seria mincomp 2–10, β = 2 (tabela, zbieżność, skład batchy, mistrzowie, front Pareto, λ/θ_D, surogat,
    źródła punktów, odległość od startu);
  - 10: start z `.gl.interior.csv` vs pełny start (mc2, mc3);
  - 11: skan β przy mc7 (β = 0, 1, 2, 5, 10);
  - 12: porównanie ze starą optymalizacją (`opt.ratios.mix/*slim.mu0.2.acqbeta2*`: tryb A, gęstość stanów
    per site, μ* = 0.2, max 0.75);
  - 13: wnioski.
- Stary `OptMix_mu0.2_Runs.ipynb` pozostawiony bez zmian.

## 2. Czy wnioski się zmieniły po przejściu na G + Lloyd? — **Nie**

Najlepsze Tc przy ≥ k pierwiastkach, nowe / stare [K]:

| k | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|
| nowe | 18.99 | 17.92 | 16.60 | 14.99 | 13.34 | 11.92 | 9.73 | 7.22 | 4.36 |
| stare | 18.96 | 18.14 | 16.90 | 15.69 | 14.06 | 12.06 | 9.66 | — | — |

- **Ten sam basen Nb–Zr.** Dla mc2–mc7 mistrzowie to praktycznie te same składy (odległość 0.01–0.09):
  Nb59Zr41 (wcześniej Nb58Zr42), Nb56Zr39Ta5, Nb56Zr34Hf5Ta5 itd.
- **Pierwiastki dodatkowe** siedzą na progu 5 %, w kolejności Ti/Ta/Hf → Y → Mo → W.
  Re i Ru pojawiają się dopiero przy mc9 i mc10.
- **λ i μ\*.** λ w GL jest niższe (1.14 zamiast 1.54 dla Nb–Zr), ale kompensuje to μ* = 0.1. θ_D jest identyczne,
  więc spadek Tc z liczbą pierwiastków idzie wyłącznie przez λ (corr(λ, Tc) = 0.998).
- **Różnice drobne:**
  - mc8: krajobraz płaski, dwa składy mają ok. 9.7 K (Nb58 + 7 dodatków oraz Nb38Zr30 + …);
  - mc10: Zr41Nb15 + 8 × 5 %;
  - Hf, Ta i W pojawiają się w mistrzach nieco częściej.
- **Optymalizator.** Przeszukiwanie praktycznie całkowicie lokalne (97–100 % punktów `local`).
- **Skan β (mc7):** β nie ma znaczenia dla wyniku (11.6–12.0 K), większe β zwiększa tylko szum batcha.
- **Ocena GL.** GL ma znaczenie dla walidacji z literaturą (MAE 1.4 vs 1.9 K), nie dla rankingu składów.
  Bezwzględne Tc nadal są zawyżone: 19 K dla Nb–Zr wobec ok. 11 K w eksperymencie.

## 3. Start `int` vs pełny start
- **Skład.** Oba starty dochodzą do tego samego składu:
  - mc2: Nb60.2Zr39.8 (18.98 K) vs Nb59.4Zr40.6 (18.99 K);
  - mc3: Nb56.9Zr37.4Ta5.7 (17.79 K) vs Nb56.1Zr38.7Ta5.1 (17.92 K).
- **Różnica 0.13 K w mc3 to szum.** W runie z pełnym startem punkt prawie identyczny ze składem mistrza `int`
  ma 17.80 K. Mistrz 17.92 K to pojedynczy punkt o ok. 0.1 K wyżej od sąsiadów.
- **Szybkość.** Zbieżność nie jest taka sama:
  - `int` potrzebuje ok. 11–21 iteracji, żeby dojść do optimum z dokładnością 1 K / 0.2 K;
  - pełny start potrzebuje 0–3 iteracji;
  - mc3 int stał 9 iteracji na plateau 17.18 K.
- **Zastrzeżenia:**
  - w mc2 pełna baza zawiera już binarne Nb67Zr33 (18.78 K), więc to nie jest uczciwy test zbieżności;
  - baza `int` jest też mniejsza (997 vs 1707 punktów), więc dwa efekty się mieszają;
  - na każdy wariant przypada jeden seed.
- **Proponowane sformułowanie:** „to samo optimum, ok. 2–5× więcej iteracji”.

## 4. Przerwane runy — przyczyna OOM
- **Limit iteracji.** `ITERATIONS=100` łącznie. `"iterations"` w `parameters.json` to liczba pozostałych iteracji
  wznawianego joba, a nie limit (np. 68 = 100 − 32).
- **Zakończone normalnie (early stop, `rc=0`):** mc2–mc5, mc8–mc10, mc7 β5, int mc2, int mc3.
  Dodatkowo mc7 β0: job 1 OOM, job 2 doliczył iterację 31 i zakończył się early stopem.
- **OOM w wszystkich 3 jobach łańcucha (`rc=137`, „oom-kill”):** mc6, mc7 (β2), mc7 β1, mc7 β10.
  Stan lokalny to ich stan końcowy (30 ukończonych iteracji). Na Topoli już nie liczą.
- **Przyczyna:** `compute_min_distances` w `scripts/run_opt_compositions.py` tworzyła pełną tablicę
  100 000 kandydatów × N znanych × 10 pierwiastków w float64, a potem drugą taką samą (`diff**2`).
  Przy N ≈ 2000 to 2 × 16 GB, a job ma `--mem=32G`.
  - Runy z pełną bazą startową (1707 punktów) przekraczały próg po ok. 29–31 iteracjach.
  - Runy `int` startowały z 997 punktów i nigdy go nie osiągnęły.
  - Kolejne joby w łańcuchu startowały już powyżej progu, więc ginęły po ok. 8 minutach, jeszcze przed KKR.
- **Skutek uboczny:** wznawianie traktowało pusty `iteration_N` jako ukończony, więc zostały dziury.
  Iteracja 30 jest pusta w mc6, β0, β1 i β10, a iteracje 31–33 są puste na końcu przerwanych runów.
- **Logi SLURM** są w `slurm/logs/opt_<tag>_<jobid>.out` (na Topoli względem `/lu/topola/home/rtop/heaml`).
  Plik `errorlog.txt` w katalogu runu nie powstał.
  Stdout Pythona był buforowany, więc log nie pokazywał ostatniego kroku procesu głównego.

## 5. Poprawki — commit `e690490` (gałąź `stoner`)
- **Odległości.** `compute_min_distances` używa `scipy.spatial.cKDTree`.
  - Wynik jak w starej wersji (różnica ≤ 2e-16 dla metryk euklidesowej i manhattan).
  - 100 000 × 2100 punktów: 585 MB i 1.5 s zamiast ok. 32 GB.
- **Wznawianie.** Nowa funkcja `iteration_has_results()`: iteracja jest zrobiona, jeśli ma `computation.tar`
  albo `computation/*/results.json`.
  - `--resume_from` usuwa końcowe iteracje bez wyników i liczy je od nowa pod tym samym numerem.
  - `topola_opt.sbatch` liczy `last` według tej samej reguły.
  - Puste iteracje w środku zostają jako dziury.
- **`PYTHONUNBUFFERED=1`** w `topola_opt.sbatch`.
- **`slurm/README.md`:** notka o OOM i komenda do ponownego wysłania.
- **Push:** z sesji zawisł (przyczyną były ustawienia VPN użytkownika); użytkownik wypchnął `e690490` sam.
  Wcześniejsze `73f6062` i `d9fbe47` są już na `origin`.

## 6. Wznowienie przerwanych runów (wysłane)
- Na Topoli: `git pull`, potem `env ONLY="mc6 mc7 mc7_b1 mc7_b10" bash slurm/submit_opt_gl_TiNbZrHfTaMoWYReRu.sh`.
  Podgląd `DRY=1` pokazał 12 poprawnych komend (4 runy × 3 joby).
- **Joby (łańcuchy po 3, `afterany`):**

  | run | job 1 (działa) | joby 2 i 3 (czekają) |
  |---|---|---|
  | mc6 | 9394285 (t10-14) | 9394286, 9394287 |
  | mc7 (β2) | 9394288 (t13-13) | 9394289, 9394290 |
  | mc7_b1 | 9394291 (t11-13) | 9394292, 9394293 |
  | mc7_b10 | 9394294 (t11-16) | 9394295, 9394296 |

- **Stan w chwili zapisu:** joby działają ponad 17 min. Poprzednio wznowione joby ginęły od OOM po ok. 8 min,
  więc to dobry znak, ale jeszcze nie potwierdzenie.
- **Kontrola:** `grep -h 'redo\|iterations \|Number of datapoints' slurm/logs/opt_mc7_b1_9394291.out`.
  Oczekiwane: `iterations 32..100` (mc7: `31..100`), linia `(IIII) Iteration: 32 ... Number of datapoints: ~2000`,
  a po niej obliczenia KKR (`lattice_*.inp`).
- **Wyniki** trafiają do tych samych katalogów `opt.ratios.gl/<NAME>/`. Log SLURM: `slurm/logs/opt_<tag>_<jobid>.out`.
- **Early stop** liczy się od iteracji, w której znaleziono mistrza (zapisanej przy punktach), więc wznowienie
  nie zeruje licznika.
- `squeue` obcina nazwę jobu do 8 znaków (`opt_mc7_` to b1 i b10). Pełne nazwy:
  `squeue -u rtop -o "%.10i %.14j %.3t %.10M %R"`.

## 7. Nowa losowa baza startowa: Re, Ru → V, Sc
- **Zestaw:** Ti, Nb, Zr, Hf, Ta, Mo, W, Y, V, Sc.
- **Projekt jak w ostatecznej bazie 10-elementowej z Re, Ru** (`databases/...ReRu...gl.csv`: 1795 punktów =
  997 wnętrze + ok. 100 na każdej ścianie k = 2…9):
  1000 punktów wnętrza simpleksu i po 100 na każdej ścianie k = 9…2, razem **1800**.
  - Na ścianach każdy obecny pierwiastek ma udział ≥ 5 %.
  - Pokrytych jest wszystkie 45 par binarnych, nie ma duplikatów.
  - W starej bazie te liczby powstały z 492 starych punktów plus uzupełnienia. Tu nie ma starych punktów z V i Sc,
    więc całość jest nowa.
- **Ziarna nowe:** wnętrze 6000…6999, ściany (70+k)·1000 + i. Przy starych ziarnach wektory udziałów byłyby
  identyczne, tylko z V i Sc na miejscu Re i Ru. Jeśli takie sparowanie byłoby potrzebne, listę można przegenerować.
- **Pliki** (commit `576dd3b`, wypchnięty):
  - `slurm/compositions/TiNbZrHfTaMoWYVSc_gl.csv` — lista; wygenerowana przez
    `python scripts/composition_list.py generate --elements Ti,Nb,Zr,Hf,Ta,Mo,W,Y,V,Sc --full 1000:6000 --face k:100:(70+k)000`
    dla k = 9…2;
  - `slurm/submit_random_gl_TiNbZrHfTaMoWYVSc.sh` — 60 tasków × 30 przeplatanych wierszy, 6 naraz, 5 rund,
    nazwa jobu `rnd10vsc`. Przy 36 taskach byłoby 50 wierszy, czyli 9 rund, co grozi limitem 2 dni;
  - wiersz w tabeli `slurm/README.md`.
- **Wysłane:** array **9394300**. 22 taski działają, 38 czeka z powodu `(Resources)`, czyli zajętego klastra,
  a nie limitu użytkownika. Całość potrwa raczej 2–3 dni.
- **Wyniki:** `random.ratios/sra.TiNbZrHfTaMoWYVSc.kp10.ew0.6.gl/`, logi `slurm/logs/rnd10vsc_9394300_<task>.out`.

## 8. Do zrobienia
- **Optymalizacje:**
  - potwierdzić w logach, że OOM nie wraca (patrz sekcja 6);
  - po zakończeniu skopiować z powrotem `opt.ratios.gl/` i `slurm/logs/`;
  - uruchomić ponownie `OptGL_mu0.1_Runs.ipynb` i zaktualizować wnioski (mc6, mc7 β1/2/10).
- **Baza V/Sc:**
  - sprawdzić, czego brakuje:
    `python scripts/composition_list.py missing slurm/compositions/TiNbZrHfTaMoWYVSc_gl.csv random.ratios/sra.TiNbZrHfTaMoWYVSc.kp10.ew0.6.gl --out slurm/compositions/TiNbZrHfTaMoWYVSc_gl.missing.csv`,
    a brakujące wysłać ponownie z `FILE=...missing.csv`;
  - skopiować wyniki na stację roboczą;
  - zbudować `.slim.gl` (`make_slim_dataset.py`) i `databases/sra.TiNbZrHfTaMoWYVSc.kp10.ew0.6.gl.csv`
    (`make_database.py`), ewentualnie też wersję `.interior`;
  - przygotować skrypt optymalizacji analogiczny do `submit_opt_gl_TiNbZrHfTaMoWYReRu.sh`.
- **Opcjonalnie:** kolumna `iters` w notebooku liczy najwyższy numer iteracji, a nie liczbę ukończonych
  (β1 i β10 pokazują 31 zamiast 30).
