#!/bin/bash
# One active-learning optimisation run (scripts/run_opt_compositions.py) on a whole Topola node, started from an
# init database CSV (databases/, scripts/make_database.py; no random.ratios needed on Topola).  Every iteration
# computes CHAMPIONS compositions with WORKERS concurrent KKR pipelines (debye_mode mix: lattice scan + final SCF +
# McMillan in the consts.py variant, GL), each specx up to all cpus of the node (OMP_THREAD_LIMIT; specx uses ~2 cores).
# Finished iterations are packed to iteration_N/computation.tar (--pack_iterations, file quota).
# Chaining: the job continues the run found in WORKDIR (--resume_from WORKDIR, ITERATIONS = total number of
# iterations) and exits at once when the run has early-stopped or reached ITERATIONS, so the same script is
# queued several times with --dependency=afterany (slurm/submit_opt_gl_TiNbZrHfTaMoWYReRu.sh).  A composition that
# was running when the wall time hit is not finished later (the next job starts a new iteration).
# Variables come from the environment (sbatch --export=ALL); never put a comma-separated value into --export=ALL,VAR=...
#SBATCH --job-name=hea_opt
#SBATCH --account=g107-2962
#SBATCH --partition=topola
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm/logs/%x_%j.out
set -uo pipefail
ROOT=/lu/topola/home/rtop/heaml
cd "${ROOT}" || exit 1
source "${ROOT}/venv/bin/activate"
NAME=${NAME:?}; INIT_DB=${INIT_DB:?}; MINCOMP=${MINCOMP:?}; BETA=${BETA:?}
OPTROOT=${OPTROOT:-opt.ratios.gl}
ELEMENTS=${ELEMENTS:-Ti,Nb,Zr,Hf,Ta,Mo,W,Y,Re,Ru}
TARGET=${TARGET:-Tc_mu0.1}; MIN_RATIO=${MIN_RATIO:-0.05}; MAX_COMP=${MAX_COMP:-0.7}; FACE=${FACE:-0.1}
ITERATIONS=${ITERATIONS:-100}; SEED=${SEED:-100}; WORKERS=${WORKERS:-10}; CHAMPIONS=${CHAMPIONS:-10}
MODELS=${MODELS:-10}
WORKDIR=${OPTROOT}/${NAME}
[ -s "${INIT_DB}" ] || { echo "init database ${INIT_DB} missing (git pull?)"; exit 2; }

if grep -qs "EARLY STOP" "${WORKDIR}/optimization_log.txt"; then
    echo "[$(date +%F_%T)] ${WORKDIR}: already early-stopped, nothing to do"; grep "EARLY STOP" "${WORKDIR}/optimization_log.txt"; exit 0
fi
# last iteration with results; trailing empty iteration_N (job killed before its compositions finished) are redone
# under the same number by run_opt_compositions.py --resume_from (same rule: computation.tar or */results.json)
last=0
for d in $(ls -d "${WORKDIR}"/iteration_* 2>/dev/null | sed 's/.*iteration_//' | sort -n); do
    if [ -f "${WORKDIR}/iteration_${d}/computation.tar" ] || ls "${WORKDIR}/iteration_${d}"/computation/*/results.json >/dev/null 2>&1; then
        last=${d}
    fi
done
remaining=$(( ITERATIONS - last ))
[ "${remaining}" -gt 0 ] || { echo "[$(date +%F_%T)] ${WORKDIR}: ${last} of ${ITERATIONS} iterations done, nothing to do"; exit 0; }
RESUME=()
[ "${last}" -gt 0 ] && RESUME=(--resume_from "${WORKDIR}")

export OMP_THREAD_LIMIT=${SLURM_CPUS_PER_TASK}       # each specx may use all cpus; WORKERS of them run at a time
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1   # see bash.old/2_run_opt_*: forked workers hung in BLAS
export PYTHONUNBUFFERED=1                            # otherwise the log loses the main process' last lines on a kill
mkdir -p "${WORKDIR}"
echo "[$(date +%F_%T)] host $(hostname)  ${WORKDIR}  init ${INIT_DB}  mincomp ${MINCOMP} (>${MIN_RATIO}) max ${MAX_COMP}  beta ${BETA}  target ${TARGET}  iterations $((last + 1))..${ITERATIONS}"
python scripts/run_opt_compositions.py \
    --workdir "${WORKDIR}/" \
    --init_db "${INIT_DB}" \
    --errorlog "${WORKDIR}/errorlog.txt" \
    ${RESUME[@]+"${RESUME[@]}"} \
    --iterations "${remaining}" \
    --workers "${WORKERS}" \
    --champions_per_step "${CHAMPIONS}" \
    --seed "${SEED}" \
    --number_of_models "${MODELS}" \
    --elements "${ELEMENTS}" \
    --min_components "${MINCOMP}" \
    --min_components_ratio "${MIN_RATIO}" \
    --max_composition "${MAX_COMP}" \
    --face_fraction "${FACE}" \
    --acquisition_beta "${BETA}" \
    --stoner_beta 0.0 \
    --debye_mode mix \
    --target "${TARGET}" \
    --pack_iterations
rc=$?
echo "[$(date +%F_%T)] ${WORKDIR} finished rc=${rc}"
exit ${rc}
