#!/bin/bash
# Shared submit step for slurm/submit_random_gl_<elements>.sh: the caller sets FILE (composition list), WORKDIR,
# NTASKS and CONCURRENT, then sources this file.  One array of NTASKS tasks of slurm/topola_complist_array.sbatch
# drains the list dynamically (rows claimed by atomic mkdir), 2-day wall time per task.  QOS (2026-10-25):
# 128 submitted / 64 running jobs per user, ~512 cpus => 37 tasks of 14 cpus run at once, so NTASKS=37 uses the
# whole allowance and every task ends at about the same time.  Rows/task ≈ rows/(NTASKS); a round of CONCURRENT
# 10-12-component alloys takes ~3-6 h.
#   env NTASKS=20 ...                   fewer tasks (longer)        env DRYRUN=1 ...   print the plan, submit nothing
#   env OFFSET=888 ...                  only rows >= 888 (the rest belongs to tasks already running)
#   env FILE=<missing.csv> RECLAIM=1 NTASKS=5 ...   finish a dataset: also take over directories without results.json
set -euo pipefail
ROOT=/lu/topola/home/rtop/heaml
cd "${ROOT}"
mkdir -p slurm/logs
[ -s "${FILE}" ] || { echo "composition list ${FILE} missing or empty"; exit 2; }
nrows=$(( $(wc -l < "${FILE}") - 1 )); OFFSET=${OFFSET:-0}; left=$(( nrows - OFFSET ))
rounds=$(( (left + NTASKS * CONCURRENT - 1) / (NTASKS * CONCURRENT) ))
printf "%s: %d rows (from row %d: %d) -> %d tasks x %d workers claiming rows (~%d rounds x 3-6 h) -> %s%s\n" \
    "${FILE}" "${nrows}" "${OFFSET}" "${left}" "${NTASKS}" "${CONCURRENT}" "${rounds}" "${WORKDIR}" "${RECLAIM:+  [reclaim stale dirs]}"
[ -n "${DRYRUN:-}" ] && exit 0
FILE="${FILE}" WORKDIR="${WORKDIR}" CONCURRENT="${CONCURRENT}" OFFSET="${OFFSET}" RECLAIM="${RECLAIM:-}" \
    sbatch --job-name="${JOBNAME:-rnd}" --array="1-${NTASKS}" ${ACCOUNT:+--account="${ACCOUNT}"} --export=ALL \
    slurm/topola_complist_array.sbatch
