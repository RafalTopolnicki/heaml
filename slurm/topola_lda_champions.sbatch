#!/bin/bash
# Topola job array: the optimisation champions (set A Tc_mu0.1, set B corr2, set B Tc_mu0.1; mincomp 2..10) recomputed
# with an LDA functional instead of PBE (2026-11-14).  Only the compositions are taken from the PBE runs: the full
# pipeline of scripts/process_hea.py runs from scratch — lattice scan (bcc, 0.95..1.05 x mixture lattice, EOS fit),
# final SCF at the new a0, McMillan-Hopfield eta (GL: gg + lloyd), mixture theta_D, Tc.  All settings except xc are the
# production ones (consts.KKR_PARAMS_*: sra, ew 0.6, mxl 3, nmag, lattice bzqlty 10, final SCF bzqlty 20, debye_mode mix).
# Compositions: rows of slurm/compositions/td_champions_gl.csv (written by scripts/td_champion_list.py); the el_* rows
# (pure-element references of the TD array) are not used.  Task i = data row ROWS[i] (set by submit_lda_champions.sh).
# Output <OUTROOT>/<run>/<dirname>/{results.json,run_params.json,kkr_raw.tar}; a row whose results.json exists is
# skipped; an unfinished directory (no results.json) is renamed to <dir>.partial_<date> and the composition redone.
# Submit with slurm/submit_lda_champions.sh.
#SBATCH --job-name=lda_cmp
#SBATCH --account=g107-2962
#SBATCH --partition=topola
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=8G
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm/logs/%x_%A_%a.out
set -uo pipefail

ROOT=/lu/topola/home/rtop/heaml
cd "${ROOT}" || exit 1
source "${ROOT}/venv/bin/activate"

LIST=${LIST:-slurm/compositions/td_champions_gl.csv}
XC=${XC:-vwn}                      # AkaiKKR LDA keywords: vwn (Vosko-Wilk-Nusair, closest to the Perdew-Wang LDA of Jaskiewicz 2019), vbh (von Barth-Hedin, Jaskiewicz 2016), mjw
OUTROOT=${OUTROOT:-lda.champions.${XC}}
ROWS=${ROWS:?ROWS not set (use slurm/submit_lda_champions.sh)}
row=$(awk -v i="${SLURM_ARRAY_TASK_ID}" '{print $i}' <<< "${ROWS}")
[ -n "${row}" ] || { echo "no row for task ${SLURM_ARRAY_TASK_ID}"; exit 1; }

# columns: tag,run,target,Tc,n_el,elements,concentrations,outdir,dirname
IFS=, read -r tag run target tc nel elements conc outdir dirname < <(sed -n "$((row + 1))p" "${LIST}")
[ -n "${tag}" ] && [ -n "${run}" ] || { echo "row ${row} of ${LIST} is not a champion row (tag '${tag}')"; exit 1; }
wd="${OUTROOT}/${run}/${dirname}"

if [ -f "${wd}/results.json" ]; then
    echo "[$(date +%F_%T)] ${tag}: ${wd}/results.json exists, nothing to do"; exit 0
fi
if [ -d "${wd}" ]; then
    mv "${wd}" "${wd}.partial_$(date +%Y%m%d_%H%M%S)" || exit 1
fi
mkdir -p "${wd}"

export OMP_THREAD_LIMIT=${SLURM_CPUS_PER_TASK}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1
echo "[$(date +%F_%T)] host $(hostname) cpus ${SLURM_CPUS_PER_TASK}  row ${row}  ${tag}  (PBE ${target} ${tc} K)  xc ${XC}  -> ${wd}"
# shellcheck disable=SC2086   (element and concentration lists are space-separated on purpose)
python scripts/process_hea.py \
    --element_labels ${elements} \
    --concentrations ${conc} \
    --workdir "${wd}" \
    --xc "${XC}" \
    --debye_mode mix \
    --task all
rc=$?
echo "[$(date +%F_%T)] ${tag}: process_hea.py exit ${rc}"
[ -f "${wd}/results.json" ] || { echo "${tag}: no results.json"; exit 3; }
exit ${rc}
