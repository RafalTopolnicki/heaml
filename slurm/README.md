# Running the KKR pipeline on Topola (ICM UW, SLURM)

Code and venv live in `/lu/topola/home/rtop/heaml`; code is exchanged only through git.
AkaiKKR `specx` path for Topola is already in `scripts/src/consts.py`.

## Node and CPU choice

Topola node: 2 × 14 cores (Xeon E5-2650 v3, Haswell), 64 or 128 GB, partition `topola`.
`process_hea.py` runs one `specx` at a time per alloy; `specx` is OpenMP (`OMP_THREAD_LIMIT`) but
does not saturate its cores, so 2–3 alloys are run concurrently on the same core set (workstation
experience: 16 alloys × 12 threads on 48 threads). Default here: half a node (14 cpus) per array
task, 3 alloys at once, each `specx` up to 14 threads, `--mem=8G` per task (specx needs little memory).
A multicomponent alloy costs ~30–70 CPU-h on the workstation; expect roughly 10–25 h wall per
task of 3 alloys on Haswell, hence `--time=2-00:00:00` (check `scontrol show partition topola`
for MaxTime and shorten if needed). Check `sacct` MaxRSS after the first tasks and raise `--mem` if needed.

## Job array (recommended): a few alloys per task

```
cd /lu/topola/home/rtop/heaml
slurm/submit_compare.sh                                  # pending entries of 0_compare_with_known_V.sh
slurm/submit_compare.sh 0_compare_with_known_bz10-10-20.sh
BINARIES=1 slurm/submit_compare.sh                       # + V binaries
PER_TASK=2 MAXPAR=6 slurm/submit_compare.sh 0_compare_with_known_V.sh 14
```

The wrapper runs the script in `LIST=1` mode (prints entries without `results.json`), groups them
`PER_TASK` per line in `slurm/logs/pending_*.txt` and submits `topola_compare_array.sbatch` with
`--array=1-N%MAXPAR`; each task runs the script with `ONLY="a b c" NJOBS=3`. Logs:
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
