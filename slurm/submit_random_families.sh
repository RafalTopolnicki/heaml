#!/bin/bash
# Shared submit loop for slurm/submit_random_gl_<elements>.sh.  The caller defines WORKDIR, ELEMENTS and
# FAMILIES=("name:k:nsamples:seed:concurrent" ...) (k empty = full simplex) and then sources this file.
# Each family becomes one job array of slurm/topola_random_array.sbatch; task = ROUNDS rounds of `concurrent`
# compositions (PER_TASK = concurrent*ROUNDS samples), 2-day wall time.  A round of 10-12-component alloys takes
# ~3-6 h on 14 cpus, so ROUNDS=6 (default) is 18-36 h per task.  QOS counts every array task as a submitted job:
# raise ROUNDS (fewer, longer tasks) if sbatch reports QOSMaxSubmitJobPerUserLimit, or submit family by family.
#   env ROUNDS=8 ...       longer tasks (24-48 h)
#   env MAXPAR=20 ...      at most 20 running tasks per array (default: no throttle)
#   env CALIB=1 ...        calibration: task 1 only of the families in CALIB_FAMILIES, a single round each
#   env DRYRUN=1 ...       print the plan, submit nothing
set -euo pipefail
ROOT=/lu/topola/home/rtop/heaml
cd "${ROOT}"
mkdir -p slurm/logs
ROUNDS=${ROUNDS:-6}; MAXPAR=${MAXPAR:-}; CALIB_FAMILIES=${CALIB_FAMILIES:-full k5 k2}
total=0; ntotal=0
for fam in "${FAMILIES[@]}"; do
    IFS=: read -r name k n seed conc <<< "${fam}"
    per=$(( conc * ROUNDS ))
    if [ -n "${CALIB:-}" ]; then
        case " ${CALIB_FAMILIES} " in *" ${name} "*) per=${conc} ;; *) continue ;; esac
        ntask=1; arr="1"
    else
        ntask=$(( (n + per - 1) / per )); arr="1-${ntask}${MAXPAR:+%${MAXPAR}}"
    fi
    total=$(( total + n )); ntotal=$(( ntotal + ntask ))
    printf "%-5s k=%-3s %4d samples  seed %-6s %d concurrent x %d rounds = %2d/task -> %3d tasks  array %s\n" \
        "${name}" "${k:-all}" "${n}" "${seed}" "${conc}" "$(( per / conc ))" "${per}" "${ntask}" "${arr}"
    [ -n "${DRYRUN:-}" ] && continue
    sbatch --job-name="rnd_${name}" --array="${arr}" ${ACCOUNT:+--account="${ACCOUNT}"} \
        --export=ALL,WORKDIR="${WORKDIR}",ELEMENTS="${ELEMENTS}",SEED="${seed}",NSAMPLES="${n}",PER_TASK="${per}",CONCURRENT="${conc}",N_NONZERO="${k}",MIN_FRAC=0.05 \
        slurm/topola_random_array.sbatch
done
echo "total ${total} compositions in ${ntotal} array tasks -> ${WORKDIR}"
