#!/bin/bash
# Re-run ONLY the final-SCF stage (scripts/finalscf.py: one SCF + DOS block -> fort.51, finalscf_*.out) for the entries
# listed in a rerun_finalscf.csv written by scripts/make_slim_dataset.py (entries whose fort.51 was deleted).
# Job array: task i takes lines (i-1)*PER_TASK+1 .. i*PER_TASK of the csv and runs them one after another on 14 cpus.
# Output: <DST>/<name>/finalscf/{fort.51, finalscf_<a0>.out, ...}; copy DST back and run make_slim_dataset.py --finish.
#   cd /lu/topola/home/rtop/heaml
#   N=`tail -n +2 random.ratios/X.slim.gl/rerun_finalscf.csv | wc -l`      # tcsh: backticks
#   sbatch --array=1-$(( (N+4)/5 ))%40 --export=ALL,CSV=random.ratios/X.slim.gl/rerun_finalscf.csv,DST=random.ratios/X.slim.gl slurm/topola_finalscf_rerun.sbatch
# A 10-component final SCF at bz20 takes ~5-15 min on 14 threads; PER_TASK=5 -> ~1 h per task.
#SBATCH --job-name=fscf_rerun
#SBATCH --account=g107-2962
#SBATCH --partition=topola
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=8G
#SBATCH --time=06:00:00
#SBATCH --output=slurm/logs/%x_%A_%a.out
set -uo pipefail
ROOT=/lu/topola/home/rtop/heaml
cd "${ROOT}" || exit 1
source "${ROOT}/venv/bin/activate"
CSV=${CSV:?path of rerun_finalscf.csv}; DST=${DST:?dataset dir that receives <name>/finalscf}
PER_TASK=${PER_TASK:-5}
export OMP_THREAD_LIMIT=${SLURM_CPUS_PER_TASK}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
first=$(( (SLURM_ARRAY_TASK_ID - 1) * PER_TASK + 2 )); last=$(( first + PER_TASK - 1 ))     # +2: skip the header
sed -n "${first},${last}p" "${CSV}" | while IFS=, read -r name elements conc a0 ew xc rel bz mxl; do
    wd="${DST}/${name}/finalscf"
    if [ -f "${wd}/fort.51" ] && ls "${wd}"/finalscf_*.out >/dev/null 2>&1; then echo "skip ${name} (done)"; continue; fi
    mkdir -p "${wd}"
    echo "[$(date +%F_%T)] ${name}  Z=${elements}  a0=${a0}  bz${bz}"
    python scripts/finalscf.py --output finalscf --elements ${elements} --concentrations ${conc} --a0 "${a0}" \
        --ew "${ew}" --xc "${xc}" --rel "${rel}" --bzqlty "${bz}" --mxl "${mxl}" --workdir "${wd}" --no-gzip > "${wd}/finalscf_rerun.log" 2>&1 \
        || echo "FAILED ${name} (see ${wd}/finalscf_rerun.log)"
done
echo "[$(date +%F_%T)] task ${SLURM_ARRAY_TASK_ID} finished"
