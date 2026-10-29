# Running the KKR pipeline on Topola (ICM UW, SLURM)

Code and venv live in `/lu/topola/home/rtop/heaml`; code is exchanged only through git.
AkaiKKR `specx` path for Topola is already in `scripts/src/consts.py`.

## Node and CPU choice

Topola node: 2 × 14 cores (Xeon E5-2650 v3, Haswell), 64 or 128 GB, partition `topola`.
`process_hea.py` runs one `specx` at a time per alloy; `specx` is OpenMP (`OMP_THREAD_LIMIT`) but
does not saturate its cores, so 2–3 alloys are run concurrently on the same core set (workstation
experience: 16 alloys × 12 threads on 48 threads). Default here: half a node (14 cpus) per array
task, 5 alloys at once (`PER_TASK=5`), each `specx` up to 14 threads, `--mem=12G` per task, `--time=12:00:00`.
A multicomponent alloy costs ~30–70 CPU-h on the workstation; expect roughly 10–25 h wall per
task of 3 alloys on Haswell, hence `--time=2-00:00:00` (check `scontrol show partition topola`
for MaxTime and shorten if needed). Check `sacct` MaxRSS after the first tasks and raise `--mem` if needed.

## Job array (recommended): a few alloys per task

```
cd /lu/topola/home/rtop/heaml
slurm/submit_compare.sh                                  # pending entries of 0_compare_with_known_V.sh
slurm/submit_compare.sh 0_compare_with_known_bz10-10-20.sh
env BINARIES=1 slurm/submit_compare.sh                   # + V binaries  (tcsh: VAR=x needs `env`)
env PER_TASK=3 MAXPAR=6 slurm/submit_compare.sh 0_compare_with_known_V.sh 14
```

The wrapper runs the script in `LIST=1` mode (prints entries without `results.json`), groups them
`PER_TASK` per line in `slurm/logs/pending_*.txt` and submits `topola_compare_array.sbatch` with
`--array=1-N%MAXPAR`; each task runs the script with `ONLY="a b c d e" NJOBS=5`. Logs:
`slurm/logs/hea_cmp_<jobid>_<task>.out` and `<workdir>/process_hea.log`.
Grant `g107-2962` is set in `#SBATCH --account=` (both `.sbatch` files); `ACCOUNT=` overrides it.

## Whole node (alternative)

```
sbatch slurm/topola_compare_node.sbatch                                   # 6 alloys × 14 threads on 28 cores
sbatch --export=ALL,SCRIPT=0_compare_with_known_bz10-10-20.sh,THREADS=28,OVERSUB=2 slurm/topola_compare_node.sbatch
```

## Other scripts

`3_structure_compare_elements_VSc.sh` and `3_run_validation_binaries_bz10-10-20.sh` honour
`NJOBS` / `OMP_THREAD_LIMIT` the same way (set them in an `sbatch --wrap` or copy the node script).
Monitoring: `squeue -u $USER`, `sacct -j <jobid> --format=JobID,State,Elapsed,MaxRSS,TotalCPU`.

## k-point convergence (final-SCF only, needs the "bigk" AkaiKKR build)

`slurm/submit_kconv.sh` submits `topola_kconv.sbatch`: a job array with one `scripts/finalscf.py` run
(full SCF + DOS/fort.51/fort.52) per bzqlty of `BZLIST`, one specx on 14 cpus, 8 GB, 12 h. Results go to
`results/test.kconv/<NAME>_<REL>_<XC>_bz<N>/` with a `composition.json`. On bcc, bzqlty 40 gives ~12 000
irreducible k-points and the stock binary stops at 10 000 (`too many k-points`), so build the patched copy
`cpa2002v010.bigk` (bzmesh.f `nk1x=400000`, spmain.f writes fort.52) next to the production one:
`${ROOT}/cpa2002v010.bigk/specx` is picked up automatically (or set `AKAIMODBIN`, or change `consts.py`;
the `AKAIMODBIN` override in `consts.py` must be committed for the automatic pick-up to work).
Rule: bzqlty is never raised above 60.

Building the patched copy on Topola (`slurm/akaikkr_bigk.patch` = the two source changes; makefile unchanged):

```
cd /lu/topola/home/rtop/heaml
cp -r cpa2002v010.potential2026 cpa2002v010.bigk
cd cpa2002v010.bigk && patch -p1 < ../slurm/akaikkr_bigk.patch
make -f makefile.gfortran clean; make -f makefile.gfortran        # (whatever built the production specx)
```

```
slurm/submit_kconv.sh                                        # Ta34Nb33Hf8Zr14Ti11 (a=3.36 A), bz 20 30 40 50 60
env BZLIST="15 25 35 45 55" slurm/submit_kconv.sh            # odd values
env NAME=Nb_a6.20 ELEMENTS=41 CONC=1.0 A0=6.20 REL=nrl XC=vbh BZLIST="20 30 40 60" slurm/submit_kconv.sh
```

Topola only computes; the analysis runs on the workstation. When the array is COMPLETED, copy the run
directories back (each holds `finalscf_*.out`, `fort.51`, `fort.52`, `composition.json`; ~0.5 MB per bzqlty)
and build the table locally:

```
rsync -av topola:/lu/topola/home/rtop/heaml/results/test.kconv/ /home/rafal/WORK/HEA/ML/results/test.kconv/
python scripts/eta_rmta_tests/17_kconv_table.py results/test.kconv/TNHZT_a3.36_sra_pbe   # workstation, repo root
```

## New random datasets for the GL re-optimisation (composition lists)

The compositions are generated **once**, on the workstation, into a CSV committed under `slurm/compositions/`
(`scripts/composition_list.py generate`; columns idx, family, seed, dirname, one per element).  SLURM tasks only
consume row ranges of that file (`run_random_compositions.py --composition_file FILE --sample_offset A
--number_of_samples N`), so nothing depends on the random generator at run time, every task is reproducible, and
what is missing is simply the rows without `results.json`:

```
python scripts/composition_list.py status  slurm/compositions/TiNbZrHfTaMoWYReRu_gl.csv random.ratios/sra.TiNbZrHfTaMoWYReRu.kp10.ew0.6.gl
python scripts/composition_list.py missing slurm/compositions/TiNbZrHfTaMoWYReRu_gl.csv random.ratios/sra.TiNbZrHfTaMoWYReRu.kp10.ew0.6.gl \
       --out slurm/compositions/TiNbZrHfTaMoWYReRu_gl.missing.csv          # then: env FILE=<that> NTASKS=5 slurm/submit_random_gl_TiNbZrHfTaMoWYReRu.sh
```

`slurm/submit_random_gl_<elements>.sh` (sets FILE, WORKDIR, NTASKS, CONCURRENT; shared step
`slurm/submit_complist.sh`) submits one array of `topola_complist_array.sbatch`: NTASKS=36 tasks × ceil(rows/NTASKS)
rows, interleaved (task i: rows i-1, i-1+36, ...; fixed rows, deliberately no dynamic claiming over the network file
system) so every task gets the same family mix (the list is sorted full → k=9 … → binaries; contiguous blocks gave
the first tasks only slow 10-component alloys), CONCURRENT=6 alloys at a time per 14-cpu task (specx uses ~2 cores
each; ~1.5 GB each of the 12G; a round of 10–12-component alloys ≈ 3–6 h, 37 rows = 7 rounds), 2-day wall time.
The cpu cap is 512 per user = 36 tasks of 14 cpus — with 37 the last one stays PENDING until another ends — so
NTASKS ≤ 36 runs everything in one wave (fewer if other jobs hold cpus); INTERLEAVE=0 gives the old contiguous blocks,
e.g. `OFFSET=n NTASKS=m` to split the rows ≥ n left over from an earlier array into m shorter tasks (array
9386076: tasks 38–55 cancelled and rows 888..1307 resubmitted as 35 tasks × 12 rows = 3 rounds).  Variables go through the environment (`--export=ALL`): a comma-separated value inside
`--export=ALL,VAR=a,b` is split by sbatch (that bug sampled pure Ti on 2026-10-25).  QOS: 128 submitted / 64 running
jobs per user, ~512 cpus (37 tasks); every array task is one job.  Full pipeline (lattice scan, Debye, final SCF,
McMillan in the `consts.py` variant, GL since 2026-10-25); `scripts/make_slim_dataset.py` recomputes eta in GL anyway.
File quota (62k files per user, venv ≈ 17k): since 2026-10-26 `process_hea.py` packs every finished composition to
`results.json`, `run_params.json`, `kkr_raw.tar` (lattice/, debye/, finalscf/; `tar -xf kkr_raw.tar` restores them;
`process_kkr`, `--reuse_from` and `make_slim_dataset.py` read the archive directly) — 4 inodes instead of ~40.
A running composition holds ≤ ~45 files, failed ones stay unpacked. Older directories:
`python scripts/cleanup_kkr_dirs.py <dir> --pack --apply` (only touches finished compositions, safe while jobs run).

| list | compositions | families (seed base) | tasks |
|---|---|---|---|
| `TiNbZrHfTaMoWYReRu_gl.csv` → `random.ratios/sra.TiNbZrHfTaMoWYReRu.kp10.ew0.6.gl` | 1308 | full 800 (2000; old set 100..1099), k9/8/7 100 each (49000/48000/47000), k6..2 top-ups 68/45/30/25/40 to 100 with the old GL-converted points (46000…42000; old 20000+1000k) | 55 |
| `TiScNbZrHfTaMoWYReRuLa_gl.csv` → `random.ratios/sra.TiScNbZrHfTaMoWYReRuLa.kp10.ew0.6.gl` | 1001 | full +1 (3000; old 999 entries, seeds 100..1099), k11…2 100 each (50000+1000k) | 42 |
| `TiNbZrHfTaMoWYVSc_gl.csv` → `random.ratios/sra.TiNbZrHfTaMoWYVSc.kp10.ew0.6.gl` (2026-10-29; Re, Ru → V, Sc; no old points, same final design as the ReRu database) | 1800 | full 1000 (6000), k9…2 100 each (70000+1000k) | 60 × 30 rows |

Generators are those of `run_random_compositions.py`: full-simplex sample i = flat Dirichlet with seed SEED+i;
k-face sample i = subset `face_subset(n, k, i, SEED)` (fixed random order of all C(n,k) subsets, so k9/k8 of the
10-element set and k11/k10/k2 of the 12-element set cover every subset) with a flat Dirichlet on it (all fractions
≥ 0.05) seeded SEED+i.  All names are unique and disjoint from the old datasets (checked at generation).
Cost ≈ 2300 compositions × 9–12 core-h ≈ 25 k core-h; with ~37 tasks running at once each list takes ~2 days.
Afterwards (workstation): rsync the `.gl` directory back and
`python scripts/make_slim_dataset.py random.ratios/<set>.gl random.ratios/<set>.slim.gl`
(merges with the entries already converted from the old set; `slim_info.json` lists both sources).

## GL optimisation runs from init databases (2026-10-28)

The optimizer no longer needs the random datasets on Topola: `run_opt_compositions.py --init_db <csv>` reads a flat
database built on the workstation by `scripts/make_database.py` and committed under `databases/` (one row per
composition, raw KKR inputs + derived Tc; derived columns are recomputed with the current code on reading, identical
to `--initdir`).  `databases/sra.TiNbZrHfTaMoWYReRu.kp10.ew0.6.gl.csv` = all 1795 GL points (1774 with θ_D),
`.gl.interior.csv` = the 997 with all 10 elements > 0.

```
bash slurm/submit_opt_gl_TiNbZrHfTaMoWYReRu.sh                          # 15 runs x CHAIN=3 chained jobs
env ONLY="mc9 mc10" bash slurm/submit_opt_gl_TiNbZrHfTaMoWYReRu.sh      # selected tags; DRY=1 prints only
```

Runs (target `Tc_mu0.1`, θ_D mix, Stoner off, max 70 %, required components > 5 %, 10 champions/iteration, seed 100,
early stop after 15 iterations without improvement): mincomp 2…10 with β=2; mincomp 7 with β=0/1/5/10; mincomp 2 and 3
from the interior database.  Output `opt.ratios.gl/<init>.mu0.1.acqbeta<β>.stonerbeta0.mixdebye.mincomp<m>.r0.05.max0.7.face0.1/`,
logs `slurm/logs/opt_<tag>_<jobid>.out`.  `topola_opt.sbatch`: whole node (28 cpus, 32G, 2 days), 10 concurrent
compositions, each specx up to 28 threads.  The same job is queued CHAIN times with `--dependency=afterany`; each one
resumes the run in place (`--resume_from` = the workdir, `ITERATIONS` = total) and exits immediately when the run has
early-stopped or is complete.  A composition interrupted by the wall time is not finished later.  15 × 28 = 420 cpus of
the ~512 cap.  File quota: `--pack_iterations` moves every finished iteration's compositions into
`iteration_N/computation.tar` (≈5 inodes per iteration instead of ~50; up to ~45 per composition while it runs);
`--resume_from` reads the archives.  Workstation, after rsync:
`for t in opt.ratios.gl/*/iteration_*/computation.tar; do tar -xf $t -C $(dirname $t)/computation; done`.
Do not create an `EXIT` file in the repo root: every run checks it and would stop.  Candidates with 9 or 10 required
elements cannot be found by rejection; `src/sampling.py` then forces random m-element subsets above the ratio
(accept probability 1/C(j, m), same uniform distribution).

**2026-10-29 — OOM fix.** mc6 and mc7 β=1/2/10 were OOM-killed (rc 137, all 3 chained jobs) once the known set reached
~2000 points: `compute_min_distances` broadcast 100k candidates × known × elements in float64 (~2 × 16 GB > 32G).
It now uses `scipy.spatial.cKDTree` (same distances to 1e-16, ~0.6 GB, 1.5 s).  A trailing `iteration_N` without
results (no `computation.tar`, no `computation/*/results.json`) is now redone under the same number on resume (both
`topola_opt.sbatch` and `--resume_from`); empty iterations in the middle stay as gaps.  `PYTHONUNBUFFERED=1` so the
log shows the main process' last step before a kill.  Resubmit:
`env ONLY="mc6 mc7 mc7_b1 mc7_b10" bash slurm/submit_opt_gl_TiNbZrHfTaMoWYReRu.sh`.

## Status log

**2026-10-23** — first array 9383051 (`0_compare_with_known_V.sh`, 7 tasks × 3 alloys, 14 cpus, 8G, 2 d)
running. Node check (`scontrol show node t5-13`): 2 × 14 physical cores, no HT, 63 GB. Each specx has
14 threads (`nlwp`) but uses only ~215 % CPU → 6 specx on a node give load ≈ 12 of 28; MaxRSS 274 MB
per task. Whether this is poor specx scaling on small lattice-stage jobs or an effect of three
processes sharing 14 cores is undecided → run `sbatch slurm/topola_specx_scaling.sbatch` (28…1
threads, exclusive node). Outcome sets `PER_TASK` (1 if specx scales, 5–6 if it saturates at 2–4
threads) and the layout of optimisation runs (10 alloys per node). Reference from c5n1: lattice-stage
run of a 5-component alloy ≈ 347 s wall at 24 threads. Compare on Topola with
`grep -h "elapsed time" results/comparison.sra.ew0.6.bz10-10-20/*/lattice/*.out`.
Practical: login shell is tcsh (use backticks or `bash`; `source venv/bin/activate.csh`);
`srun --jobid` needs the numeric id from `squeue -j <array> -r -o "%A %K %N"`; `sstat -j <id>_<i>.batch`
works; ssh to nodes needs a key. 2-day limit may be tight for 3 slow alloys per task; an interrupted
alloy restarts from scratch, finished ones are skipped on resubmit.

**2026-10-23, after the run** — all 7 tasks COMPLETED in 1:30–2:02 h (`sacct`). Average cores used
of 14: 5.5 for 3 five/six-component alloys, 3.6 for a single ternary alloy alone → specx scales poorly
even alone, packing pays (3 alloys ≈ 1.3 × the wall time of 1). MaxRSS 4.35 GB for 3 alloys.
Cost ≈ 9 allocated core-h per 5-component alloy (the 30–70 CPU-h of the grant estimate came from the
unreliable AkaiKKR `cpu(sec)` line). Defaults changed: `PER_TASK=5`, `--mem=12G`, `--time=12:00:00`.
`sacct` TotalCPU / (Elapsed × cpus) is the reliable utilisation measure.
