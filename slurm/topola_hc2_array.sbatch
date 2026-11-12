#!/bin/bash
# Topola job array: Hc2 (AkaiKKR A(k,E) -> D, rho0, dHc2/dT) on the baseline set.  NOT a Tc run:
# everything is written under hc2_runs/ (never results/).  Each task takes one line of LISTFILE (PER_TASK system
# dirs) and runs scripts/hc2_spc/run_hc2_pipeline.sh on them CONCURRENTLY (specx does not saturate its cores).
# Submit with slurm/submit_hc2_baseline.sh.
#SBATCH --job-name=hc2
#SBATCH --account=g107-2962
#SBATCH --partition=topola
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=12G
#SBATCH --time=12:00:00
#SBATCH --output=slurm/logs/%x_%A_%a.out
set -uo pipefail

ROOT=/lu/topola/home/rtop/heaml
cd "${ROOT}" || exit 1
source "${ROOT}/venv/bin/activate"
export PYTHON=python
export AKAIMODBIN=${AKAIMODBIN:-${ROOT}/cpa2002v010.potential2026/specx}

LISTFILE=${LISTFILE:?set by submit_hc2_baseline.sh}
dirs=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "${LISTFILE}")
[ -n "${dirs}" ] || { echo "no line ${SLURM_ARRAY_TASK_ID} in ${LISTFILE}"; exit 1; }

export OMP_THREAD_LIMIT=${SLURM_CPUS_PER_TASK}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

echo "[$(date +%F_%T)] host $(hostname) cpus ${SLURM_CPUS_PER_TASK} specx ${AKAIMODBIN}: ${dirs}"
for d in ${dirs}; do
    scripts/hc2_spc/run_hc2_pipeline.sh "${d}" > /dev/null 2>&1 &
done
wait
for d in ${dirs}; do echo "$(cat "${d}/hc2_status.txt" 2>/dev/null || echo MISSING)  ${d}"; done
echo "[$(date +%F_%T)] finished"
