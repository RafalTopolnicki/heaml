#!/bin/bash
# Shared submit step for slurm/submit_random_gl_<elements>.sh: the caller sets FILE (composition list), WORKDIR,
# NTASKS (array size) and CONCURRENT, then sources this file.  Rows per task = ceil(rows / NTASKS); a task is
# PER_TASK/CONCURRENT rounds of CONCURRENT alloys (a round of 10-12-component alloys takes ~3-6 h on 14 cpus),
# 2-day wall time.  QOS (checked 2026-10-25): 128 submitted / 64 running jobs per user, 512 cpus per user = 36 tasks
# of 14 cpus (a 37th stays PENDING until one ends), so NTASKS <= 36 runs everything in one wave — fewer if other
# jobs of yours hold cpus.  Every array task counts as one job.  Rows are interleaved (task i: rows i-1, i-1+N, ...)
# so all tasks get the same family mix and end together; INTERLEAVE=0 gives contiguous blocks.
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
per=$(( (left + NTASKS - 1) / NTASKS ))
if [ "${INTERLEAVE:-1}" = 1 ]; then ntask=$(( left < NTASKS ? left : NTASKS )); stride=${ntask}
else ntask=$(( (left + per - 1) / per )); stride=1; fi
printf "%s: %d rows (from row %d: %d) -> %d tasks x <=%d rows, stride %d (%d concurrent, %d rounds of <=%d h) -> %s\n" \
    "${FILE}" "${nrows}" "${OFFSET}" "${left}" "${ntask}" "${per}" "${stride}" "${CONCURRENT}" "$(( (per + CONCURRENT - 1) / CONCURRENT ))" 6 "${WORKDIR}"
[ -n "${DRYRUN:-}" ] && exit 0
FILE="${FILE}" WORKDIR="${WORKDIR}" PER_TASK="${per}" CONCURRENT="${CONCURRENT}" OFFSET="${OFFSET}" STRIDE="${stride}" \
    sbatch --job-name="${JOBNAME:-rnd}" --array="1-${ntask}" ${ACCOUNT:+--account="${ACCOUNT}"} --export=ALL \
    slurm/topola_complist_array.sbatch
