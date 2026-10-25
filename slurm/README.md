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

## New random dataset for the GL re-optimisation (TiNbZrHfTaMoWYReRu)

`slurm/submit_random_gl_TiNbZrHfTaMoWYReRu.sh` submits one job array of `topola_random_array.sbatch` per family into
`random.ratios/sra.TiNbZrHfTaMoWYReRu.kp10.ew0.6.gl` (full pipeline: lattice scan, Debye, final SCF, McMillan;
the eta variant does not matter here, it is recomputed in GL by `scripts/make_slim_dataset.py` afterwards):

| family | samples | seed base | per task | tasks |
|---|---|---|---|---|
| full simplex (k=10) | 800 | 2000 (old set: 100..1099) | 4 | 200 |
| k = 9 / 8 / 7 faces | 100 each | 49000 / 48000 / 47000 | 4 | 25 each |
| k = 6 / 5 / 4 / 3 / 2 faces (top-up to 100 with the 32/55/70/75/60 old GL-converted points) | 68 / 45 / 30 / 25 / 40 | 46000 … 42000 (old: 20000+1000k) | 5 | 14 / 9 / 6 / 5 / 8 |

Seeds: every task of a family shares `SEED`; sample j (global index, `--sample_offset`) uses seed `SEED+j`, so
parallel tasks never repeat a composition and the k-face subset order is common to all tasks (full C(10,k)
coverage).  All new seed ranges are disjoint from the ones used for the old dataset.
Cost: a 14-cpu task of 4–5 compositions takes ~3–6 h (10-component alloys ≈ 2× a 5-component one; 12 h limit);
≈ 1300 compositions = ~300 tasks ≈ 15–20 k core-h.  Wall time < 2 days needs ≥ 30 tasks running at once
(`MAXPAR`, default 15 per array × 9 arrays).  Run `env CALIB=1 slurm/submit_random_gl_TiNbZrHfTaMoWYReRu.sh` first (task 1 of the
k=10, k=5, k=2 arrays) and read the elapsed time in `slurm/logs/rnd_*_<jobid>_1.out` before the full submission.
Afterwards (workstation): rsync the `.gl` directory back and
`python scripts/make_slim_dataset.py random.ratios/sra.TiNbZrHfTaMoWYReRu.kp10.ew0.6.gl random.ratios/sra.TiNbZrHfTaMoWYReRu.kp10.ew0.6.slim.gl`
(merges with the 492 entries already converted from the old set; `slim_info.json` lists both sources).

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
