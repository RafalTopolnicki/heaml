#!/bin/bash
# k-point convergence of one composition on Topola: a job array in which task i runs scripts/finalscf.py
# (full SCF + DOS/fort.51 block) with the i-th bzqlty of BZLIST.  Needs the "bigk" AkaiKKR build
# (bzmesh.f nk1x 400000, fort.52 detl) for bzqlty > 30 on bcc: bz40 gives ~12 000 irreducible k-points,
# the stock binary stops at 10 000.  Submit with slurm/submit_kconv.sh (sets the composition and BZLIST).
#
# Env (all optional except via the wrapper): NAME (dir prefix), ELEMENTS ("73 41 ..."), CONC ("0.34 0.33 ..."),
# A0 (a.u.), REL (sra|nrl|srals), XC (pbe|vbh), EW, MXL, BZLIST ("20 30 40 50 60"), OUT (results/test.kconv),
# AKAIMODBIN (explicit specx; default ${ROOT}/cpa2002v010.bigk/specx if present, else consts.py path).
# One task = one specx on 14 cpus; pure V bz60 took ~15 min on 14 threads locally, a 5-component CPA alloy
# is a few times slower per k-point -> 12 h is a safe limit.  Never use bzqlty > 60 (user rule).
#SBATCH --job-name=kconv
#SBATCH --account=g107-2962
#SBATCH --partition=topola
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --output=slurm/logs/%x_%A_%a.out
set -uo pipefail

ROOT=/lu/topola/home/rtop/heaml
cd "${ROOT}" || exit 1
source "${ROOT}/venv/bin/activate"

NAME=${NAME:-TNHZT_a3.36}
ELEMENTS=${ELEMENTS:-"73 41 72 40 22"}
CONC=${CONC:-"0.34 0.33 0.08 0.14 0.11"}
A0=${A0:-6.34949}
REL=${REL:-sra}; XC=${XC:-pbe}; EW=${EW:-0.6}; MXL=${MXL:-3}
BZLIST=${BZLIST:-"20 30 40 50 60"}
OUT=${OUT:-results/test.kconv}
bz=$(awk -v i="${SLURM_ARRAY_TASK_ID}" '{print $i}' <<< "${BZLIST}")
[ -n "${bz}" ] || { echo "no bzqlty #${SLURM_ARRAY_TASK_ID} in BZLIST='${BZLIST}'"; exit 1; }
[ "${bz}" -le 60 ] || { echo "bzqlty ${bz} > 60 refused"; exit 1; }

if [ -z "${AKAIMODBIN:-}" ] && [ -x "${ROOT}/cpa2002v010.bigk/specx" ]; then
    export AKAIMODBIN="${ROOT}/cpa2002v010.bigk/specx"
fi
export OMP_THREAD_LIMIT=${SLURM_CPUS_PER_TASK}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

WD="${OUT}/${NAME}_${REL}_${XC}_bz${bz}"
mkdir -p "${WD}"
python - "${WD}" ${ELEMENTS} -- ${CONC} <<'PY'
import json, sys
a = sys.argv[2:]; i = a.index('--')
json.dump({'anclr': [int(z) for z in a[:i]], 'conc': [float(c) for c in a[i+1:]]}, open(sys.argv[1] + '/composition.json', 'w'))
PY
echo "[$(date +%F_%T)] host $(hostname)  cpus ${SLURM_CPUS_PER_TASK}  ${NAME} bzqlty ${bz}  specx ${AKAIMODBIN:-consts.py default}  -> ${WD}"
python scripts/finalscf.py --output finalscf --elements ${ELEMENTS} --concentrations ${CONC} --a0 "${A0}" \
    --ew "${EW}" --xc "${XC}" --rel "${REL}" --bzqlty "${bz}" --mxl "${MXL}" --workdir "${WD}" --no-gzip
rc=$?
grep -h "elapsed time\|too many k-points\|nk=" "${WD}"/finalscf_*.out 2>/dev/null | sort -u | head -5
echo "[$(date +%F_%T)] finished ${NAME} bz${bz} rc=${rc}"
exit ${rc}
