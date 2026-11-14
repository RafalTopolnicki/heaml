#!/bin/bash
# Topola job array (2026-11-14): does the energy window ew = 0.6 Ry also spoil the Tc of Re-containing alloys?
# Pure bcc Re is broken at ew 0.6 (valence band bottom below the contour; ew 1.0 and 1.3 agree to 0.3 mRy).
# Full process_hea.py pipeline (PBE, lattice scan + EOS, final SCF, eta GL, theta_D mix, Tc) with the production
# settings except ew:
#   task 0  A_mu0.1_mc9   ew 1.0   (production ew 0.6 value = optimisation champion, td_champions_gl.csv)
#   task 1  A_mu0.1_mc10  ew 1.0   (idem)
#   task 2  Mo60Re40      ew 0.6   (no GL ew 0.6 result exists for Mo-Re; exp. Tc 12-13 K near Mo60Re40)
#   task 3  Mo60Re40      ew 1.0
# Output re_ew_tc/<tag>.ew<ew>/<composition>/results.json (git-ignored); a task whose results.json exists is skipped,
# an unfinished directory is renamed to <dir>.partial_<date>.
# Submit (login node):  sbatch slurm/topola_re_ew_tc.sbatch      (or --array=2,3 for a subset)
#SBATCH --job-name=re_ew_tc
#SBATCH --account=g107-2962
#SBATCH --partition=topola
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=8G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-3
#SBATCH --output=slurm/logs/%x_%A_%a.out
set -uo pipefail

ROOT=/lu/topola/home/rtop/heaml
cd "${ROOT}" || exit 1
source "${ROOT}/venv/bin/activate"

# tag | elements | concentrations | ew | dirname
TASKS=(
  "A_mu0.1_mc9|Ti Nb Zr Hf Ta Mo W Y Re|0.056311 0.445136 0.141964 0.082100 0.067832 0.051005 0.052911 0.051216 0.051526|1.0|Ti0.0563Nb0.4451Zr0.1420Hf0.0821Ta0.0678Mo0.0510W0.0529Y0.0512Re0.0515"
  "A_mu0.1_mc10|Ti Nb Zr Hf Ta Mo W Y Re Ru|0.058058 0.148916 0.413325 0.056683 0.064732 0.050705 0.050814 0.054258 0.051518 0.050993|1.0|Ti0.0581Nb0.1489Zr0.4133Hf0.0567Ta0.0647Mo0.0507W0.0508Y0.0543Re0.0515Ru0.0510"
  "Mo60Re40|Mo Re|0.6 0.4|0.6|Mo0.6000Re0.4000"
  "Mo60Re40|Mo Re|0.6 0.4|1.0|Mo0.6000Re0.4000"
)
entry=${TASKS[${SLURM_ARRAY_TASK_ID}]:-}
[ -n "${entry}" ] || { echo "no task ${SLURM_ARRAY_TASK_ID}"; exit 1; }
IFS='|' read -r tag elements conc ew dirname <<< "${entry}"
wd="re_ew_tc/${tag}.ew${ew}/${dirname}"

if [ -f "${wd}/results.json" ]; then
    echo "[$(date +%F_%T)] ${tag} ew ${ew}: ${wd}/results.json exists, nothing to do"; exit 0
fi
if [ -d "${wd}" ]; then
    mv "${wd}" "${wd}.partial_$(date +%Y%m%d_%H%M%S)" || exit 1
fi
mkdir -p "${wd}"

export OMP_THREAD_LIMIT=${SLURM_CPUS_PER_TASK}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1
echo "[$(date +%F_%T)] host $(hostname) cpus ${SLURM_CPUS_PER_TASK}  ${tag}  ew ${ew}  -> ${wd}"
# shellcheck disable=SC2086   (element and concentration lists are space-separated on purpose)
python scripts/process_hea.py \
    --element_labels ${elements} \
    --concentrations ${conc} \
    --workdir "${wd}" \
    --ew "${ew}" \
    --debye_mode mix \
    --task all
rc=$?
echo "[$(date +%F_%T)] ${tag} ew ${ew}: process_hea.py exit ${rc}"
[ -f "${wd}/results.json" ] || { echo "${tag}: no results.json"; exit 3; }
exit ${rc}
