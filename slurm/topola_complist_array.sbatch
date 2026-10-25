#!/bin/bash
# Compute a composition list (CSV from scripts/composition_list.py, committed under slurm/compositions/) with the
# full KKR pipeline (lattice scan + Debye + final SCF + McMillan) as a SLURM job array that drains the list
# dynamically: every task runs CONCURRENT workers, each claims the next free row by creating its directory (atomic
# mkdir), computes it and takes the next one (run_random_compositions.py --claim).  So any number of tasks share one
# list and all finish at about the same time — no idle second wave; tasks that find nothing left exit at once.
# Rows with results.json are skipped; a directory without results.json belongs to a running (or killed) task —
# after everything finished, `python scripts/composition_list.py missing FILE WORKDIR --out F2` and a resubmission
# with RECLAIM=1 recompute those.  OFFSET = first row considered (rows below it belong to other tasks).
# Submit with slurm/submit_random_gl_<elements>.sh (sets FILE, WORKDIR, CONCURRENT, OFFSET, RECLAIM in the
# environment; never put a comma-separated value inside --export=ALL,VAR=..., sbatch splits it at commas).
#SBATCH --job-name=hea_rnd
#SBATCH --account=g107-2962
#SBATCH --partition=topola
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=12G
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm/logs/%x_%A_%a.out
set -uo pipefail
ROOT=/lu/topola/home/rtop/heaml
cd "${ROOT}" || exit 1
source "${ROOT}/venv/bin/activate"
FILE=${FILE:?}; WORKDIR=${WORKDIR:?}; CONCURRENT=${CONCURRENT:-4}; OFFSET=${OFFSET:-0}; RECLAIM=${RECLAIM:-}
[ -s "${FILE}" ] || { echo "composition list ${FILE} missing or empty (git pull?)"; exit 2; }
nrows=$(( $(wc -l < "${FILE}") - 1 ))
export OMP_THREAD_LIMIT=${SLURM_CPUS_PER_TASK}       # each specx may use all cpus; CONCURRENT of them run at a time
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
mkdir -p "${WORKDIR}"
echo "[$(date +%F_%T)] host $(hostname)  ${FILE} (${nrows} rows, from row ${OFFSET}) -> ${WORKDIR}  ${CONCURRENT} workers claiming rows${RECLAIM:+ (reclaim stale)}"
python scripts/run_random_compositions.py --workdir "${WORKDIR}" --errorlog "${WORKDIR}/errorlog.txt" \
    --composition_file "${FILE}" --claim --sample_offset "${OFFSET}" --workers "${CONCURRENT}" ${RECLAIM:+--reclaim}
echo "[$(date +%F_%T)] task ${SLURM_ARRAY_TASK_ID} finished rc=$?"
