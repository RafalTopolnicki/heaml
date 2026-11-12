#!/bin/bash
# Hc2 on the baseline set (Topola).  Inputs come from scripts/hc2_spc/baseline_bundle.json (committed; built on the
# workstation by make_baseline_bundle.py), run dirs go to hc2_runs/<SET>/<name>/ -- deliberately OUTSIDE results/,
# so they can never be mistaken for Tc runs.  Systems whose hc2_status.txt says DONE are not resubmitted;
# FAILED/RUNNING ones resume from their last finished step.
#
#   slurm/submit_hc2_baseline.sh                       # all pending systems, 5 per task, 14 cpus
#   env ONLY="HfNbTa Ta34Nb33Hf8Zr14Ti11" slurm/submit_hc2_baseline.sh      # test on a few systems
#   env PER_TASK=3 MAXPAR=10 slurm/submit_hc2_baseline.sh
#   env DRY=1 slurm/submit_hc2_baseline.sh              # only make inputs + list, no sbatch
set -euo pipefail
ROOT=/lu/topola/home/rtop/heaml
cd "${ROOT}"
source "${ROOT}/venv/bin/activate"
SET=${SET:-baseline.sra.ew0.6.bz10-10-20.gl}
OUT=hc2_runs/${SET}
PER_TASK=${PER_TASK:-5}
MAXPAR=${MAXPAR:-20}
CPUS=${CPUS:-14}
mkdir -p slurm/logs "${OUT}"
[ -f hc2_runs/README.txt ] || cat > hc2_runs/README.txt <<'TXT'
Hc2 / Maki estimates from AkaiKKR A(k,E) (scripts/hc2_spc/, branch rtop/hc2).  NOT Tc results.
Per system: fs/fs_summary.json (final, |v| from cone LSQ), fs/fs_summary_normals.json, star/*_results.csv,
hc2_status.txt, hc2_pipeline.log.  Raw spc runs packed in fs/fs_runs.tar.gz, star/star_runs.tar.gz.
TXT

python -c "import numpy, scipy" || { echo "venv needs numpy + scipy"; exit 1; }
python scripts/hc2_spc/make_spc_inputs.py --bundle scripts/hc2_spc/baseline_bundle.json --outdir "${OUT}" \
       --skip_existing ${ONLY:+--only ${ONLY}} > /dev/null

LISTFILE="slurm/logs/pending_hc2_$(date +%Y%m%d_%H%M%S).txt"
for d in "${OUT}"/*/; do
    d=${d%/}; n=$(basename "${d}")
    if [ -n "${ONLY:-}" ] && ! grep -qw -- "${n}" <<< "${ONLY}"; then continue; fi
    [ "$(cat "${d}/hc2_status.txt" 2>/dev/null)" = DONE ] && continue
    echo "${d}"
done | xargs -n "${PER_TASK}" > "${LISTFILE}" || true
N=$(wc -l < "${LISTFILE}")
if [ "${N}" -eq 0 ]; then echo "nothing pending in ${OUT}"; rm -f "${LISTFILE}"; exit 0; fi
echo "$(wc -w < "${LISTFILE}") systems in ${N} tasks x ${PER_TASK} (${LISTFILE})"
[ -n "${DRY:-}" ] && { cat "${LISTFILE}"; exit 0; }
sbatch --array="1-${N}%${MAXPAR}" --cpus-per-task="${CPUS}" ${ACCOUNT:+--account="${ACCOUNT}"} \
       --export=ALL,LISTFILE="${LISTFILE}" slurm/topola_hc2_array.sbatch
