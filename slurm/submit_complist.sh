#!/bin/bash
# Shared submit step for slurm/submit_random_gl_<elements>.sh: the caller sets FILE (composition list), WORKDIR,
# NTASKS (array size) and CONCURRENT, then sources this file.  Rows per task = ceil(rows / NTASKS); a task is
# PER_TASK/CONCURRENT rounds of CONCURRENT alloys (a round of 10-12-component alloys takes ~3-6 h on 14 cpus),
# 2-day wall time.  QOS (checked 2026-10-25): 128 submitted / 64 running jobs per user, ~512 cpus (37 tasks);
# every array task counts as one job.
#   env NTASKS=30 ...   fewer, longer tasks         env DRYRUN=1 ...   print the plan, submit nothing
#   env OFFSET=888 NTASKS=35 ...   split only rows >= 888 (rows below belong to an array already running)
#   env FILE=... ...    e.g. a `composition_list.py missing` file to finish a dataset (same WORKDIR)
set -euo pipefail
ROOT=/lu/topola/home/rtop/heaml
cd "${ROOT}"
mkdir -p slurm/logs
[ -s "${FILE}" ] || { echo "composition list ${FILE} missing or empty"; exit 2; }
nrows=$(( $(wc -l < "${FILE}") - 1 )); OFFSET=${OFFSET:-0}; left=$(( nrows - OFFSET ))
[ "${left}" -gt 0 ] || { echo "OFFSET ${OFFSET} >= ${nrows} rows: nothing to submit"; exit 2; }
per=$(( (left + NTASKS - 1) / NTASKS )); ntask=$(( (left + per - 1) / per ))
printf "%s: %d rows (from row %d: %d) -> %d tasks x %d rows (%d concurrent, %d rounds of <=%d h) -> %s\n" \
    "${FILE}" "${nrows}" "${OFFSET}" "${left}" "${ntask}" "${per}" "${CONCURRENT}" "$(( (per + CONCURRENT - 1) / CONCURRENT ))" 6 "${WORKDIR}"
[ -n "${DRYRUN:-}" ] && exit 0
FILE="${FILE}" WORKDIR="${WORKDIR}" PER_TASK="${per}" CONCURRENT="${CONCURRENT}" OFFSET="${OFFSET}" \
    sbatch --job-name="${JOBNAME:-rnd}" --array="1-${ntask}" ${ACCOUNT:+--account="${ACCOUNT}"} --export=ALL \
    slurm/topola_complist_array.sbatch
