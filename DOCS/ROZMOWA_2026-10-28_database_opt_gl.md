# Rozmowa 2026-10-28 — scalenie zbiorów GL, baza CSV, optymalizacje GL na Topoli

Commity na `stoner`: `73f6062` (baza CSV, `--init_db`, skrypty .sh wyjęte z gita), `d9fbe47` (SLURM dla optymalizacji,
pakowanie iteracji, poprawka samplera). Na dzień zapisu **nie wypchnięte** (`git push` przed `git pull` na Topoli).
Wcześniej użytkownik sam zacommitował `d239107` (m.in. poprawka `cleanup_kkr_dirs.py`).

## 1. Wyniki z Topoli i scalenie z danymi starymi
- `random.ratios/sra.TiNbZrHfTaMoWYReRu.kp10.ew0.6.gl` (lista `slurm/compositions/TiNbZrHfTaMoWYReRu_gl.csv`):
  1303/1308 policzone, 5 nieudanych (errorlog.txt) — **nie dokańczamy**. Wszystkie 1308 w run_params: `gg` + `lloyd`
  (metoda G z normalizacją Lloyda), co potwierdzają logi sweepu McMillana.
- Stary `.slim.gl` (492 wpisy): run_params mówią `valence` (oryginalne obliczenia 30.09–20.10), ale η/λ/Tc przeliczono
  w GL 25.10 przez `make_slim_dataset.py` (`eta_variant = gg+lloyd`). 700 dalszych starych wpisów bez fort.51
  (`rerun_finalscf.csv`) **nie wchodzi** do zbioru (metoda A tylko w surowym katalogu `...ew0.6`).
- Kontrole: `make_slim_dataset` na nowych danych = results.json z Topoli (rel. ≤1e-13, dG ≤4e-8); parametry KKR
  identyczne (10/10/20, ew 0.6, pbe, sra, Debye kkr); RandomForest wytrenowany na nowych → średnie residua starych ≈ 0
  (η, Tc, θ_D, a0) — brak systematycznego przesunięcia.
- **`.slim.gl` = 1795 wpisów** (492 + 1303), po 4 pliki; loader optymalizacji bierze 1774 (21 bez θ_D z KKR).
  `slim_info.json` ma sekcję `parts` (part1/part2).
- Rozkład 1795 wg liczby zerowych składników: wnętrze sympleksu (k=10) 997; k=9,8,7,6,5,3 po 100; k=4 i k=2 po 99.

## 2. Pełne archiwum `.gl`
- `random.ratios/sra.TiNbZrHfTaMoWYReRu.kp10.ew0.6.gl` = te same 1795 nazw co `.slim.gl`, każdy wpis
  `results.json` + `run_params.json` + `kkr_raw.tar` (1,7 GB), opis w `README_ARCHIVE.md`.
- 492 stare wpisy skopiowane z `...ew0.6`: results.json i mcmillan_cutoff_results.csv w wersji GL; oryginały wariantu A
  w tarze jako `finalscf/*.variantA.*`; run_params ustawione na gg/lloyd + klucz `eta_recomputed`; .pot usunięte,
  spakowane jak na Topoli. Weryfikacja: `make_slim_dataset` z tych tarów odtwarza slim bit w bit.
- Wpisom z Topoli uzupełniono `name` i `eta_variant` w results.json.
- 5 nieudanych katalogów → `random.ratios/sra.TiNbZrHfTaMoWYReRu.kp10.ew0.6.gl.failed`.
- Przy pakowaniu znaleziony błąd `cleanup_kkr_dirs.py` (podwójne usuwanie .pot, gdy są X.inp i X.inp.gz) — poprawiony.

## 3. Baza CSV zamiast katalogów (limit plików na Topoli)
- `scripts/make_database.py SRC... --out databases/X.csv` → jeden wiersz na skład: surowe dane (skład, θ_D, stałe
  sprężyste, B0, a0, gęstość, masa, energia, η i N(E_F) per pierwiastek/kanał, parametry KKR) + pochodne (η_mix, λ,
  Tc_mu*, Tc_sf, Stoner, dG, cechy) + `element_labels`, `source`, `eta_variant`.
- `src/database.py` (`read_database`): kolumny pochodne są odrzucane i liczone od nowa aktualnym kodem
  (`derive_kkr`) — zmiany w consts działają bez przebudowy bazy. `process_kkr` rozdzielony na `read_kkr_raw` +
  `derive_kkr` (wynik bez zmian).
- `run_opt_compositions.py --init_db a.csv [b.csv]` zamiast `--initdir` (jedno z dwóch). Weryfikacja: identyczne 1774
  wiersze bit w bit vs skanowanie katalogu; próba `--iterations 0 --debye_mode mix` daje identyczne
  init_tc_comparison.csv (≤2.5e-14 K).
- Bazy (w gicie, `databases/` nie jest ignorowane):
  `databases/sra.TiNbZrHfTaMoWYReRu.kp10.ew0.6.gl.csv` (1795 wierszy, 274 kolumny, 10 MB) i
  `databases/sra.TiNbZrHfTaMoWYReRu.kp10.ew0.6.gl.interior.csv` (997 wierszy, wszystkie 10 pierwiastków > 0).
- `notebooks/Optimization_Progress.ipynb` (poza gitem) czyta też `init_db` z parameters.json.
- `.slim.gl` nie jest już potrzebny do optymalizacji (zostaje lokalnie).

## 4. Porządki
- Użytkownik przeniósł skrypty `*.sh` z katalogu głównego do `bash.old/` (od teraz tylko SLURM). 9 śledzonych
  skryptów usunięto z gita (`git rm --cached`), `/bash.old/` w `.gitignore`. Nowe skrypty → `slurm/`.

## 5. Optymalizacje GL na Topoli
`slurm/submit_opt_gl_TiNbZrHfTaMoWYReRu.sh` (uruchamiać `bash ...`, nie sbatch) + `slurm/topola_opt.sbatch`:

| tag | baza | mincomp | β |
|---|---|---|---|
| mc2 … mc10 | `.gl.csv` | 2…10 | 2 |
| mc7_b0, mc7_b1, mc7_b5, mc7_b10 | `.gl.csv` | 7 | 0/1/5/10 |
| int_mc2, int_mc3 | `.gl.interior.csv` | 2/3 | 2 |

- Wspólne: `--target Tc_mu0.1`, `--debye_mode mix`, `--stoner_beta 0` (kara Stonera nadal jest w kodzie),
  `--max_composition 0.7`, `--min_components_ratio 0.05`, `--face_fraction 0.1`, 10 champions, 10 modeli, seed 100,
  maks. **100 iteracji łącznie** (`ITERATIONS`), early stop 15. Wynik `opt.ratios.gl/<baza>.mu0.1.acqbeta<β>.stonerbeta0.mixdebye.mincomp<m>.r0.05.max0.7.face0.1/`
  (gitignored), logi `slurm/logs/opt_<tag>_<jobid>.out`.
- Zasoby: cały węzeł 28 cpu, 32G, 2 dni, 10 składów naraz (OMP_THREAD_LIMIT=28); 15 × 28 = 420 cpu (< ~512).
- Łańcuch: każdy przebieg CHAIN=3 razy (`--dependency=afterany`), kolejne zadanie wznawia w miejscu
  (`--resume_from` = workdir, liczy tylko brakujące do 100) albo kończy od razu po EARLY STOP / komplecie.
  Skład przerwany limitem czasu nie jest dokańczany.
- `--pack_iterations`: zakończona iteracja → `iteration_N/computation.tar` (~5 i-węzłów/iterację zamiast ~50;
  bez tego 15 przebiegów ≈ 27 tys. i-węzłów przy ~25 tys. wolnych). Lokalnie przed notebookami:
  `for t in opt.ratios.gl/*/iteration_*/computation.tar; do tar -xf $t -C $(dirname $t)/computation; done`
- Sampler (`src/sampling.py`): dla mincomp 9/10 przy 5 % losowanie z odrzucaniem padało („Could only generate N global
  candidates”). Teraz dolne granice wbudowane w próbkę, a gdy odrzucanie nie wystarcza — wymuszony losowy podzbiór m
  pierwiastków z akceptacją 1/C(j, m) (Monte Carlo: ten sam rozkład). mincomp 2–8 bez zmian.
- CatBoost: `allow_writing_files=False` (bez wspólnego `catboost_info/`).
- Nie tworzyć pliku `EXIT` w katalogu głównym repo (zatrzymałby wszystkie przebiegi); pojedynczy przebieg → `scancel`.

## Otwarte
- `git push`, potem na Topoli: `git pull`, `env DRY=1 bash slurm/submit_opt_gl_TiNbZrHfTaMoWYReRu.sh`,
  `bash slurm/submit_opt_gl_TiNbZrHfTaMoWYReRu.sh`, `squeue -u $USER`.
- Po 2–3 iteracjach sprawdzić czas iteracji w logach; jeśli > ~3 h, zwiększyć CHAIN.
- Na Topoli można usunąć `random.ratios/*.gl` po upewnieniu się, że kopia jest lokalnie (optymalizacja ich nie potrzebuje).
- Nie przetestowano realnej iteracji z KKR w nowym układzie (tylko odczyt, wznawianie, sampler).
