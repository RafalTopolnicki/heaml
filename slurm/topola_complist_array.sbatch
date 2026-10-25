#!/bin/bash
# Compute a composition list (CSV from scripts/composition_list.py, committed under slurm/compositions/) with the
# full KKR pipeline (lattice scan + Debye + final SCF + McMillan) as a SLURM job array.  Task i computes rows
# [(i-1)*PER_TASK, i*PER_TASK) of FILE into WORKDIR, CONCURRENT of them at a time on its 14 cpus (the process pool
# starts the next row as soon as one finishes).  Rows whose results.json exists are skipped, so a task that hit the
# wall time is resubmitted as is, and `python scripts/composition_list.py missing FILE WORKDIR --out F2` gives the
# list of what is left.  Submit with slurm/submit_random_gl_<elements>.sh (sets FILE, WORKDIR, PER_TASK, CONCURRENT
# in the environment; never put a comma-separated value inside --export=ALL,VAR=..., sbatch splits it at commas).
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
FILE=${FILE:?}; WORKDIR=${WORKDIR:?}; PER_TASK=${PER_TASK:?}; CONCURRENT=${CONCURRENT:-4}
[ -s "${FILE}" ] || { echo "composition list ${FILE} missing or empty (git pull?)"; exit 2; }
nrows=$(( $(wc -l < "${FILE}") - 1 ))
offset=$(( (SLURM_ARRAY_TASK_ID - 1) * PER_TASK ))
n=$(( nrows - offset )); [ "${n}" -gt "${PER_TASK}" ] && n=${PER_TASK}
[ "${n}" -gt 0 ] || { echo "task ${SLURM_ARRAY_TASK_ID}: nothing to do (offset ${offset} >= ${nrows} rows)"; exit 0; }
[ "${CONCURRENT}" -gt "${n}" ] && CONCURRENT=${n}
export OMP_THREAD_LIMIT=${SLURM_CPUS_PER_TASK}       # each specx may use all cpus; CONCURRENT of them run at a time
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
mkdir -p "${WORKDIR}"
echo "[$(date +%F_%T)] host $(hostname)  ${FILE} (${nrows} rows) -> ${WORKDIR}  rows ${offset}..$((offset + n - 1))  (${CONCURRENT} concurrent)"
python scripts/run_random_compositions.py --workdir "${WORKDIR}" --errorlog "${WORKDIR}/errorlog.txt" \
    --composition_file "${FILE}" --sample_offset "${offset}" --number_of_samples "${n}" --workers "${CONCURRENT}"
echo "[$(date +%F_%T)] task ${SLURM_ARRAY_TASK_ID} finished rc=$?"
