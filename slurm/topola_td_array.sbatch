#!/bin/bash
# Topola job array: bcc / fcc / hcp total-energy comparison (scripts/structure_compare.py) for the compositions of
# LIST (slurm/compositions/td_champions_gl.csv, written by scripts/td_champion_list.py on the workstation).
# Task i = data row i of LIST (1-based, header excluded).  A row whose <outdir>/<dirname>/structure_summary.json
# exists is skipped.  structure_compare.py has no restart: an interrupted composition is redone from scratch.
# KKR settings = consts.KKR_PARAMS_LATTICE (sra, pbe, ew 0.6, bzqlty 10, mxl 3, nmag) — the same as the lattice stage
# of the optimisation runs and as all earlier thermodynamic/ results (structure_compare.py takes no overrides here).
# 49 SCF runs per composition (bcc 7, fcc 7, hcp 5 c/a x 7), one specx at a time.
# Submit with slurm/submit_td_champions.sh (only the pending rows).
#SBATCH --job-name=td_cmp
#SBATCH --account=g107-2962
#SBATCH --partition=topola
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=8G
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm/logs/%x_%A_%a.out
set -uo pipefail

ROOT=/lu/topola/home/rtop/heaml
cd "${ROOT}" || exit 1
source "${ROOT}/venv/bin/activate"

LIST=${LIST:-slurm/compositions/td_champions_gl.csv}
ROWS=${ROWS:-}                     # optional: space-separated data-row numbers; task i takes the i-th of them
row=${SLURM_ARRAY_TASK_ID}
[ -n "${ROWS}" ] && row=$(awk -v i="${SLURM_ARRAY_TASK_ID}" '{print $i}' <<< "${ROWS}")
[ -n "${row}" ] || { echo "no row for task ${SLURM_ARRAY_TASK_ID}"; exit 1; }

# columns: tag,run,target,Tc,n_el,elements,concentrations,outdir,dirname
IFS=, read -r tag run target tc nel elements conc outdir dirname < <(sed -n "$((row + 1))p" "${LIST}")
[ -n "${tag}" ] || { echo "no data row ${row} in ${LIST}"; exit 1; }

if [ -f "${outdir}/${dirname}/structure_summary.json" ]; then
    echo "[$(date +%F_%T)] ${tag}: ${outdir}/${dirname}/structure_summary.json exists, nothing to do"; exit 0
fi

export OMP_THREAD_LIMIT=${SLURM_CPUS_PER_TASK}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1
mkdir -p "${outdir}"
echo "[$(date +%F_%T)] host $(hostname) cpus ${SLURM_CPUS_PER_TASK}  row ${row}  ${tag}  ${target} ${tc} K  -> ${outdir}/${dirname}"
# shellcheck disable=SC2086   (element and concentration lists are space-separated on purpose)
python scripts/structure_compare.py \
    --element_labels ${elements} \
    --concentrations ${conc} \
    --outdir "${outdir}"
rc=$?
echo "[$(date +%F_%T)] ${tag}: structure_compare.py exit ${rc}"
[ -f "${outdir}/${dirname}/structure_summary.json" ] || { echo "${tag}: no structure_summary.json"; exit 3; }
exit ${rc}
