#!/bin/bash
# Compute a composition list (CSV from scripts/composition_list.py, committed under slurm/compositions/) with the
# full KKR pipeline (lattice scan + Debye + final SCF + McMillan) as a SLURM job array.  With STRIDE=N (the array
# size; default of submit_complist.sh) task i computes rows OFFSET+i-1, OFFSET+i-1+N, ... — every task gets the same
# mix of the list's families (the list is sorted full simplex -> k=9 ... -> binaries, so contiguous blocks give the
# first tasks all the slow 10-component alloys).  Without STRIDE task i computes rows [(i-1)*PER_TASK, i*PER_TASK)
# above OFFSET.  CONCURRENT rows at a time on its 14 cpus (the process pool starts the next row as soon as one finishes).
# Finished compositions are packed by process_hea.py to results.json + run_params.json + kkr_raw.tar (file quota).  Rows whose results.json exists are skipped, so a task that hit the
# wall time is resubmitted as is, and `python scripts/composition_list.py missing FILE WORKDIR --out F2` gives the
# list of what is left.  Submit with slurm/submit_random_gl_<elements>.sh (sets FILE, WORKDIR, PER_TASK, CONCURRENT
# and OFFSET in the environment; never put a comma-separated value inside --export=ALL,VAR=..., sbatch splits it at commas).
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
FILE=${FILE:?}; WORKDIR=${WORKDIR:?}; PER_TASK=${PER_TASK:?}; CONCURRENT=${CONCURRENT:-4}; OFFSET=${OFFSET:-0}
STRIDE=${STRIDE:-1}
[ -s "${FILE}" ] || { echo "composition list ${FILE} missing or empty (git pull?)"; exit 2; }
nrows=$(( $(wc -l < "${FILE}") - 1 ))
if [ "${STRIDE}" -gt 1 ]; then                                     # interleaved rows
    offset=$(( OFFSET + SLURM_ARRAY_TASK_ID - 1 ))
    n=$(( (nrows - offset + STRIDE - 1) / STRIDE ))
else                                                               # contiguous block
    offset=$(( OFFSET + (SLURM_ARRAY_TASK_ID - 1) * PER_TASK ))    # rows below OFFSET belong to another array
    n=$(( nrows - offset ))
fi
[ "${n}" -gt "${PER_TASK}" ] && n=${PER_TASK}
[ "${n}" -gt 0 ] || { echo "task ${SLURM_ARRAY_TASK_ID}: nothing to do (offset ${offset} >= ${nrows} rows)"; exit 0; }
[ "${CONCURRENT}" -gt "${n}" ] && CONCURRENT=${n}
export OMP_THREAD_LIMIT=${SLURM_CPUS_PER_TASK}       # each specx may use all cpus; CONCURRENT of them run at a time
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
mkdir -p "${WORKDIR}"
echo "[$(date +%F_%T)] host $(hostname)  ${FILE} (${nrows} rows) -> ${WORKDIR}  rows ${offset}..$((offset + (n - 1) * STRIDE)) step ${STRIDE} (${n} rows, ${CONCURRENT} concurrent)"
python scripts/run_random_compositions.py --workdir "${WORKDIR}" --errorlog "${WORKDIR}/errorlog.txt" \
    --composition_file "${FILE}" --sample_offset "${offset}" --sample_stride "${STRIDE}" --number_of_samples "${n}" \
    --workers "${CONCURRENT}"
echo "[$(date +%F_%T)] task ${SLURM_ARRAY_TASK_ID} finished rc=$?"
