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
