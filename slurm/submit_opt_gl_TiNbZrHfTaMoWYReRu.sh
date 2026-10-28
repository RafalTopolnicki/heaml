#!/bin/bash
# GL optimisation runs on set B (Ti,Nb,Zr,Hf,Ta,Mo,W,Y,Re,Ru), 2026-10-28: target Tc_mu0.1 (GL eta, mu* 0.1),
# theta_D mix, Stoner penalty off, max 70 % per element, required components >= 5 %, 10 champions per iteration,
# early stop after 15 iterations without improvement (consts.EARLY_STOP_PATIENCE), seed 100 everywhere.
#   * mincomp 2..10, beta 2                     init databases/sra.TiNbZrHfTaMoWYReRu.kp10.ew0.6.gl.csv (1795, 1774 usable)
#   * mincomp 7, beta 0 / 1 / 5 / 10            same init (beta 2 is the run above)
#   * mincomp 2 and 3, beta 2                   init .gl.interior.csv (997, all 10 elements > 0)
# Each run = one whole-node job (28 cpus, 10 concurrent compositions), queued CHAIN times with
# --dependency=afterany: a later job resumes the run and exits at once if it has already early-stopped.
# 15 runs x 28 cpus = 420 of the ~512-cpu user cap.  Run on the login node with bash, NOT with sbatch:
#   bash slurm/submit_opt_gl_TiNbZrHfTaMoWYReRu.sh                     all 15 runs, CHAIN=3 (3 x 2 days)
#   env ONLY="mc7_b0 mc7_b10" bash slurm/submit_opt_gl_TiNbZrHfTaMoWYReRu.sh     selected tags
#   env DRY=1 bash slurm/submit_opt_gl_TiNbZrHfTaMoWYReRu.sh           print what would be submitted
# Stop a run cleanly: scancel its jobs (an EXIT file in the repo root would stop ALL runs at the next check).
[ -n "${SLURM_JOB_ID:-}" ] && { echo "this submit script calls sbatch itself: run it on the login node WITHOUT sbatch, e.g.  bash slurm/submit_opt_gl_TiNbZrHfTaMoWYReRu.sh"; exit 2; }
cd "$(dirname "$0")/.." || exit 1
CHAIN=${CHAIN:-3}; ONLY=${ONLY:-}; DRY=${DRY:-0}
DB_ALL=databases/sra.TiNbZrHfTaMoWYReRu.kp10.ew0.6.gl.csv
DB_INT=databases/sra.TiNbZrHfTaMoWYReRu.kp10.ew0.6.gl.interior.csv
# tag  init-db  mincomp  beta
RUNS="
mc2      ${DB_ALL} 2  2
mc3      ${DB_ALL} 3  2
mc4      ${DB_ALL} 4  2
mc5      ${DB_ALL} 5  2
mc6      ${DB_ALL} 6  2
mc7      ${DB_ALL} 7  2
mc8      ${DB_ALL} 8  2
mc9      ${DB_ALL} 9  2
mc10     ${DB_ALL} 10 2
mc7_b0   ${DB_ALL} 7  0
mc7_b1   ${DB_ALL} 7  1
mc7_b5   ${DB_ALL} 7  5
mc7_b10  ${DB_ALL} 7  10
int_mc2  ${DB_INT} 2  2
int_mc3  ${DB_INT} 3  2
"
mkdir -p slurm/logs
n=0
while read -r tag db mc beta; do
    [ -z "${tag}" ] && continue
    if [ -n "${ONLY}" ] && [[ " ${ONLY} " != *" ${tag} "* ]]; then continue; fi
    [ -s "${db}" ] || { echo "missing ${db} (git pull?)"; exit 2; }
    NAME=$(basename "${db}" .csv).mu0.1.acqbeta${beta}.stonerbeta0.mixdebye.mincomp${mc}.r0.05.max0.7.face0.1
    export NAME INIT_DB=${db} MINCOMP=${mc} BETA=${beta}
    prev=""
    for ((k = 1; k <= CHAIN; k++)); do
        dep=(); [ -n "${prev}" ] && dep=(--dependency=afterany:${prev})
        if [ "${DRY}" = 1 ]; then
            echo "sbatch --job-name=opt_${tag} ${dep[*]} slurm/topola_opt.sbatch   NAME=${NAME} INIT_DB=${db} MINCOMP=${mc} BETA=${beta}"
            prev="<${tag}_${k}>"
        else
            prev=$(sbatch --parsable --export=ALL --job-name="opt_${tag}" "${dep[@]}" slurm/topola_opt.sbatch) || exit 3
            echo "${tag}: job ${prev} (${k}/${CHAIN})  -> opt.ratios.gl/${NAME}"
        fi
    done
    n=$((n + 1))
done <<< "${RUNS}"
echo "${n} runs x ${CHAIN} chained jobs"
