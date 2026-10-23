#!/bin/bash
# Topola (ICM UW) alternative to the job array: ONE whole node (28 cores) running several alloys of a
# 0_compare_with_known*.sh script concurrently, exactly like on the old workstation.
#   sbatch slurm/topola_compare_node.sbatch                                  # 0_compare_with_known_V.sh
#   sbatch --export=ALL,SCRIPT=0_compare_with_known_bz10-10-20.sh slurm/topola_compare_node.sbatch
#   sbatch --export=ALL,THREADS=28,OVERSUB=2 slurm/topola_compare_node.sbatch  # 2 alloys x 28 threads
# NJOBS = OVERSUB * cpus / THREADS (default 3 * 28 / 14 = 6 alloys at a time, each specx up to 14
# threads): specx does not saturate the cores, so 2-3 concurrent runs per core set give better
# throughput.  Use the array version when many alloys are pending: it packs better in the queue and
# a node failure costs a few alloys, not all.
#SBATCH --job-name=hea_cmp_node
#SBATCH --account=g107-2962
#SBATCH --partition=topola
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=16G
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm/logs/%x_%j.out
set -uo pipefail

ROOT=/lu/topola/home/rtop/heaml
cd "${ROOT}" || exit 1
source "${ROOT}/venv/bin/activate"
mkdir -p slurm/logs

SCRIPT=${SCRIPT:-0_compare_with_known_V.sh}
THREADS=${THREADS:-14}
OVERSUB=${OVERSUB:-3}
export OMP_THREAD_LIMIT=${THREADS}
export NJOBS=$(( OVERSUB * SLURM_CPUS_PER_TASK / THREADS ))
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

echo "[$(date +%F_%T)] host $(hostname)  cpus ${SLURM_CPUS_PER_TASK}  NJOBS ${NJOBS} x ${THREADS} threads  script ${SCRIPT}"
./"${SCRIPT}"
echo "[$(date +%F_%T)] finished ${SCRIPT}"
