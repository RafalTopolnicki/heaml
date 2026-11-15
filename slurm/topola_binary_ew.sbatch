#!/bin/bash
# Topola job array (2026-11-15): Tc of the second-maximum binaries Mo-Re and Mo-Ru as a function of the energy window.
# Questions:
#   (1) is the KKR-CPA underestimate of Re-rich alloys (old result: Mo60Re40 ~0 K vs exp. 12-13 K) an artefact of the
#       production window ew = 0.6 Ry, which does not enclose the valence band bottom of pure Re (needs >= 0.9) and W
#       (>= 0.75)?
#   (2) is ew = 0.7 Ry, the widest window at which Hf still converges (4f level enters the contour at >= 0.75), enough
#       for these alloys (E_F of Mo-Re lies lower above the band bottom than in pure Re)?
#   (3) control: Nb60Zr40 (first maximum) must not depend on the window.
# Full scripts/process_hea.py pipeline (PBE, lattice scan + EOS, final SCF, eta GL, theta_D mix, Tc), production
# settings except --ew.  Grid: ew in EWS x compositions in COMPS (defaults below: 4 x 10 = 40 tasks).
# Output binary_ew/ew<ew>/<composition>/results.json (git-ignored); finished tasks are skipped, unfinished directories
# renamed to <dir>.partial_<date>.
# Submit on the login node:   sbatch slurm/topola_binary_ew.sbatch
#   subset (task = i_ew * n_comp + i_comp):   sbatch --array=0-9 slurm/topola_binary_ew.sbatch   (ew 0.6 only)
#SBATCH --job-name=bin_ew
#SBATCH --account=g107-2962
#SBATCH --partition=topola
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=8G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-39
#SBATCH --output=slurm/logs/%x_%A_%a.out
set -uo pipefail

ROOT=/lu/topola/home/rtop/heaml
cd "${ROOT}" || exit 1
source "${ROOT}/venv/bin/activate"

EWS=(0.6 0.7 0.8 1.0)
# elements|concentrations|dirname
COMPS=(
  "Mo Re|0.9 0.1|Mo0.9000Re0.1000"
  "Mo Re|0.8 0.2|Mo0.8000Re0.2000"
  "Mo Re|0.7 0.3|Mo0.7000Re0.3000"
  "Mo Re|0.6 0.4|Mo0.6000Re0.4000"
  "Mo Re|0.5 0.5|Mo0.5000Re0.5000"
  "Mo Ru|0.9 0.1|Mo0.9000Ru0.1000"
  "Mo Ru|0.8 0.2|Mo0.8000Ru0.2000"
  "Mo Ru|0.7 0.3|Mo0.7000Ru0.3000"
  "Mo Ru|0.6 0.4|Mo0.6000Ru0.4000"
  "Nb Zr|0.6 0.4|Nb0.6000Zr0.4000"
)
NC=${#COMPS[@]}
i=${SLURM_ARRAY_TASK_ID}
[ "${i}" -lt $(( ${#EWS[@]} * NC )) ] || { echo "task ${i} outside the grid"; exit 1; }
ew=${EWS[$(( i / NC ))]}
IFS='|' read -r elements conc dirname <<< "${COMPS[$(( i % NC ))]}"
wd="binary_ew/ew${ew}/${dirname}"

if [ -f "${wd}/results.json" ]; then
    echo "[$(date +%F_%T)] ${dirname} ew ${ew}: results.json exists, nothing to do"; exit 0
fi
if [ -d "${wd}" ]; then
    mv "${wd}" "${wd}.partial_$(date +%Y%m%d_%H%M%S)" || exit 1
fi
mkdir -p "${wd}"

export OMP_THREAD_LIMIT=${SLURM_CPUS_PER_TASK}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1
echo "[$(date +%F_%T)] host $(hostname) cpus ${SLURM_CPUS_PER_TASK}  task ${i}  ${dirname}  ew ${ew}  -> ${wd}"
# shellcheck disable=SC2086   (element and concentration lists are space-separated on purpose)
python scripts/process_hea.py \
    --element_labels ${elements} \
    --concentrations ${conc} \
    --workdir "${wd}" \
    --ew "${ew}" \
    --debye_mode mix \
    --task all
rc=$?
echo "[$(date +%F_%T)] ${dirname} ew ${ew}: process_hea.py exit ${rc}"
[ -f "${wd}/results.json" ] || { echo "${dirname} ew ${ew}: no results.json"; exit 3; }
exit ${rc}
