#!/bin/bash
# Topola (ICM UW) job array for the 0_compare_with_known*.sh scripts.  Each array task takes one line
# of LISTFILE (PER_TASK alloy names, default 3) and runs them CONCURRENTLY on its cpus with
# ONLY="a b c" NJOBS=3 -- specx does not saturate the cores, so 2-3 specx on the same cores give
# better throughput than one (workstation experience).  Submit with slurm/submit_compare.sh.
#
# Topola node: 2 x 14 cores (Xeon E5-2650 v3), 64 or 128 GB.  Default 14 cpus (half a node) per task,
# 3 alloys x 14 OpenMP threads each on those 14 cores; 8 GB per task (specx needs little memory).
#SBATCH --job-name=hea_cmp
#SBATCH --account=g107-2962
#SBATCH --partition=topola
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=8G
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm/logs/%x_%A_%a.out
set -uo pipefail

ROOT=/lu/topola/home/rtop/heaml
cd "${ROOT}" || exit 1
source "${ROOT}/venv/bin/activate"

SCRIPT=${SCRIPT:-0_compare_with_known_V.sh}
LISTFILE=${LISTFILE:?set by submit_compare.sh (one line per task, several entry names per line)}
names=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "${LISTFILE}")
[ -n "${names}" ] || { echo "no line ${SLURM_ARRAY_TASK_ID} in ${LISTFILE}"; exit 1; }
n=$(wc -w <<< "${names}")

export OMP_THREAD_LIMIT=${SLURM_CPUS_PER_TASK}     # each specx may use all cpus of the task
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

echo "[$(date +%F_%T)] host $(hostname)  cpus ${SLURM_CPUS_PER_TASK}  script ${SCRIPT}  ${n} concurrent: ${names}"
NJOBS="${n}" ONLY="${names}" ./"${SCRIPT}"
echo "[$(date +%F_%T)] finished: ${names}"
