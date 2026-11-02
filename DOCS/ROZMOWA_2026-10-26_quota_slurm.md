# Rozmowa 2026-10-26 — quota plików na Topoli, pakowanie wyników KKR, nowy podział tablicy SLURM

## Problem
Po uruchomieniu zadań na Topoli pojawiło się przekroczenie quoty na liczbę plików. Limit to 62k plików:
venv zajmuje ok. 16,7k, inne pliki ok. 20k, a jedna policzona kompozycja HEA to ok. 40 i-węzłów (lattice/, debye/, finalscf/).

## Rozwiązanie: pakowanie do `kkr_raw.tar` (commit cf38749)
- `process_hea.py`: po kroku McMillana katalogi `lattice/`, `debye/` i `finalscf/` (też `finalscf.copy`) trafiają do
  `<kompozycja>/kkr_raw.tar` (`src.utils.pack_kkr_files`). To zwykły, nieskompresowany tar: duże pliki i tak są już
  .gz, a tryb dopisywania działa. Na dysku zostaje tylko **katalog + results.json + run_params.json + kkr_raw.tar
  = 4 i-węzły**. Flaga `--no-pack` wyłącza pakowanie. Po błędzie pliki zostają niespakowane, do debugowania.
- `src.utils.read_kkr_file(compdir, rel)` czyta plik z dysku, a gdy go nie ma, z archiwum (ostatnia kopia wygrywa). Korzystają z tego:
  `process_kkr` (debye_summary.json, mcmillan_cutoff_results.csv), `--reuse_from` (lattice_summary.txt,
  debye_summary.json), `make_slim_dataset.py` (fort.51, finalscf_*.out wprost z tar-a) oraz notebooki
  `Tc_Baseline_Comparison*.ipynb` (poza gitem, zmienione lokalnie).
- `scripts/cleanup_kkr_dirs.py ROOT --pack [--apply]` pakuje już policzone katalogi. Rusza tylko kompozycje z
  results.json i run_params.json, więc można go puszczać w trakcie obliczeń.
- `tar -xf kkr_raw.tar` w katalogu kompozycji przywraca oryginalny układ (sprawdzone, fort.51.gz identyczny bajt w bajt).
- Pliki .out z lattice/ i debye/ nie są po obliczeniach do niczego potrzebne. `finalscf_*.out` i `fort.51` są potrzebne
  tylko do ponownego liczenia η (make_slim_dataset w wariancie GL+Lloyd, macmillan_opw, eta_rmta_tests), dlatego są pakowane, a nie kasowane.
- Testy na kopiach MoRe/Mo60Re40 i Mo80Re20: process_kkr daje to samo dla wersji spakowanej i niespakowanej,
  `--reuse_from` działa ze spakowanego źródła, `make_slim_dataset` daje to samo results.json (CSV różni się tylko kolumną ze ścieżką).

## Podział tablicy SLURM (commit cf38749)
- Przyczyna zadania wiszącego w kolejce: limit 512 CPU na użytkownika daje 36 zadań po 14 CPU, a było NTASKS=55, więc nadmiarowe czekały.
- Teraz **NTASKS=36**, a wiersze są przeplatane (`STRIDE`, `run_random_compositions.py --sample_stride`): zadanie i liczy
  wiersze i−1, i−1+36, … Każde zadanie ma 36–37 wierszy, w tym ~22 stopy 10-składnikowe (średnio k≈8,6).
  Wcześniej ciągłe bloki dawały pierwszym zadaniom same wolne stopy.
- **CONCURRENT=6** zamiast 4: 7 rund zamiast 10, ok. 21–42 h przy limicie 48 h. To szacunek: zakładam, że specx używa ~2 rdzeni
  na stop, a ~1,5 GB × 6 mieści się w 12G.
- Wybór wierszy z pliku composition_list według pozycji, nie kolumny idx. To naprawia listy `missing`.
- `INTERLEAVE=0` przywraca stary podział na ciągłe bloki.

## Szacunek i-węzłów (lista 10-pierwiastkowa, 1308 kompozycji)
- Po zakończeniu: 1308 × 4 + ok. 80 (parameters*.json, errorlog, logi) ≈ **5,3 tys.**
- Szczyt w trakcie: 216 kompozycji liczonych naraz × ≤45 plików, czyli + ok. 10 tys., razem **≈ 15 tys.** Wolnego miejsca jest ok. 25 tys.
- Każda nieudana kompozycja zostaje niespakowana (ok. 40 plików).

## Uruchamianie (zapadła decyzja: na razie tylko 10 pierwiastków)
Skrypt `submit_random_gl_*.sh` uruchamia się **bez sbatch**, bo sam wywołuje sbatch. Trzy próby przez `sbatch`
(9393325–27) skończyły się od razu i niczego nie policzyły. Od commitów 8d4dc30 i 68c9ab8 skrypt uruchomiony pod sbatch kończy się komunikatem.
```
cd ~/heaml
git pull
rm slurm-939332*.out
bash slurm/submit_random_gl_TiNbZrHfTaMoWYReRu.sh      # ma wypisać plan i "Submitted batch job ..."
squeue -u $USER                                        # 36 x rnd10, stan R
```
Przed uruchomieniem: stary katalog `random.ratios/sra.TiNbZrHfTaMoWYReRu.kp10.ew0.6.gl` trzeba spakować
(`cleanup_kkr_dirs.py ... --pack --apply`, policzone kompozycje są pomijane) albo usunąć.
Postęp: `python scripts/composition_list.py status slurm/compositions/TiNbZrHfTaMoWYReRu_gl.csv random.ratios/sra.TiNbZrHfTaMoWYReRu.kp10.ew0.6.gl`.
Logi są w `slurm/logs/rnd10_<JOBID>_<i>.out`.
Dokończenie braków: `composition_list.py missing ... --out ..._gl.missing.csv`, potem `env FILE=<ten plik> NTASKS=5 bash slurm/submit_random_gl_TiNbZrHfTaMoWYReRu.sh`.
Na końcu skopiować katalog rsync-iem na stację roboczą i uruchomić `make_slim_dataset.py <.gl> <.slim.gl>`.

## Otwarte
- Na dzień zapisu tablica nie była jeszcze poprawnie wysłana (czekam na wynik `bash slurm/submit_...` i `squeue`).
- Lista 12-pierwiastkowa: później, po zakończeniu 10-pierwiastkowej. Obie naraz to 72 zadania.
- Lokalnie niezacommitowana jest jednolinijkowa zmiana w `scripts/src/consts.py` (pusta linia), nie z tej rozmowy.
- Stare notebooki `Initial_CollectResults` i `*CollectResults-Copy1` nadal czytają debye_summary.json wprost z dysku.
