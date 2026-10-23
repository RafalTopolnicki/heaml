#!/bin/bash
# Submit the pending entries of a 0_compare_with_known*.sh script to Topola as a SLURM job array
# (slurm/topola_compare_array.sbatch).  Each array task runs PER_TASK alloys concurrently on CPUS cores
# (specx does not saturate the cores, so 2-3 concurrent runs per core set pay off).
#
#   slurm/submit_compare.sh                                   # 0_compare_with_known_V.sh, 14 cpus, 3 alloys/task
#   slurm/submit_compare.sh 0_compare_with_known_bz10-10-20.sh 14
#   PER_TASK=2 slurm/submit_compare.sh                        # 2 alloys per task
#   BINARIES=1 slurm/submit_compare.sh                        # also the V binaries (env is passed through)
#   MAXPAR=6 slurm/submit_compare.sh                          # at most 6 array tasks running at once
#   ACCOUNT=G12-34 slurm/submit_compare.sh                    # override #SBATCH --account in the .sbatch
#
# Entries with an existing results.json are not submitted (LIST=1 mode of the script).
set -euo pipefail
ROOT=/lu/topola/home/rtop/heaml
cd "${ROOT}"
SCRIPT=${1:-0_compare_with_known_V.sh}
CPUS=${2:-14}
PER_TASK=${PER_TASK:-3}
MAXPAR=${MAXPAR:-20}
[ -x "${SCRIPT}" ] || { echo "no such script: ${SCRIPT}"; exit 1; }
mkdir -p slurm/logs

LISTFILE="slurm/logs/pending_$(basename "${SCRIPT}" .sh)_$(date +%Y%m%d_%H%M%S).txt"
LIST=1 NJOBS=1 ./"${SCRIPT}" | grep -vE '^(===|\[|Using )' | sort -u | xargs -n "${PER_TASK}" > "${LISTFILE}" || true
N=$(wc -l < "${LISTFILE}")
if [ "${N}" -eq 0 ]; then echo "nothing pending for ${SCRIPT}"; rm -f "${LISTFILE}"; exit 0; fi
echo "$(wc -w < "${LISTFILE}") pending entries in ${N} tasks x ${PER_TASK} (${LISTFILE}):"; cat "${LISTFILE}"

sbatch --array="1-${N}%${MAXPAR}" --cpus-per-task="${CPUS}" \
       ${ACCOUNT:+--account="${ACCOUNT}"} \
       --export=ALL,SCRIPT="${SCRIPT}",LISTFILE="${LISTFILE}" \
       slurm/topola_compare_array.sbatch
