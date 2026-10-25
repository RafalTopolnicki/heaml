#!/bin/bash
# Random / face compositions of a full KKR dataset (scripts/run_random_compositions.py: lattice scan + Debye + final
# SCF + McMillan) as a SLURM job array.  Task i computes the global samples [(i-1)*PER_TASK, i*PER_TASK) of one
# family (full simplex, or the k-faces with N_NONZERO=k), concurrently on its 14 cpus.  All tasks share SEED:
# sample j uses seed SEED+j, so seeds are unique across tasks and, for faces, the C(n,k) subset order is common.
# Submit with slurm/submit_random_gl.sh (sets WORKDIR, ELEMENTS, SEED, NSAMPLES, PER_TASK, N_NONZERO, MIN_FRAC).
#SBATCH --job-name=hea_rnd
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
WORKDIR=${WORKDIR:?}; ELEMENTS=${ELEMENTS:?}; SEED=${SEED:?}; NSAMPLES=${NSAMPLES:?}
PER_TASK=${PER_TASK:-5}; N_NONZERO=${N_NONZERO:-}; MIN_FRAC=${MIN_FRAC:-0.05}
offset=$(( (SLURM_ARRAY_TASK_ID - 1) * PER_TASK ))
n=$(( NSAMPLES - offset )); [ "${n}" -gt "${PER_TASK}" ] && n=${PER_TASK}
[ "${n}" -gt 0 ] || { echo "task ${SLURM_ARRAY_TASK_ID}: nothing to do (offset ${offset} >= ${NSAMPLES})"; exit 0; }
export OMP_THREAD_LIMIT=${SLURM_CPUS_PER_TASK}       # each specx may use all cpus; n of them run concurrently
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
mkdir -p "${WORKDIR}"
echo "[$(date +%F_%T)] host $(hostname)  ${WORKDIR}  family=${N_NONZERO:-full}  seed ${SEED}  samples ${offset}..$((offset + n - 1))  (${n} concurrent)"
python scripts/run_random_compositions.py --workdir "${WORKDIR}" --errorlog "${WORKDIR}/errorlog.txt" --elements "${ELEMENTS}" \
    --number_of_samples "${n}" --sample_offset "${offset}" --seed "${SEED}" --workers "${n}" \
    ${N_NONZERO:+--n_nonzero "${N_NONZERO}" --min_nonzero_fraction "${MIN_FRAC}"}
echo "[$(date +%F_%T)] task ${SLURM_ARRAY_TASK_ID} finished rc=$?"
