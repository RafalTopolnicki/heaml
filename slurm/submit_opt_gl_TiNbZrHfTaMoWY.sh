#!/bin/bash
# GL optimisation runs on set C = Ti,Nb,Zr,Hf,Ta,Mo,W,Y (8 elements, no Re/Ru/V/Sc), mincomp 2 ... 8 (2026-11-18).
# mincomp 8 = all elements of the set (mincomp 9, 10 do not exist for 8 elements).
# Init database databases/sra.TiNbZrHfTaMoWY.kp10.ew0.6.gl.csv (1200 random compositions computed from scratch:
# 600 full simplex + 100 faces per k = 7..2; slurm/submit_random_gl_TiNbZrHfTaMoWY.sh; 1179 usable, thetaDB present).
# Settings identical to the set-A runs (submit_opt_gl_TiNbZrHfTaMoWYReRu.sh) and the uncorrected set-B runs: target
# Tc_mu0.1 (no corrections), GL eta, theta_D mix, mu* 0.1, Stoner penalty off, max 70 % per element, required
# components >= 5 %, beta 2, 10 champions per iteration, early stop after 15 iterations, seed 100.
# Output opt.ratios.gl/sra.TiNbZrHfTaMoWY.kp10.ew0.6.gl.mu0.1.acqbeta2.stonerbeta0.mixdebye.mincomp<m>.r0.05.max0.7.face0.1,
# logs slurm/logs/opt_<tag>_<jobid>.out.  7 runs x 28 cpus = 196 cpus.  Run on the login node with bash, NOT with sbatch:
#   bash slurm/submit_opt_gl_TiNbZrHfTaMoWY.sh                          all 7 runs, CHAIN=3 (3 x 2 days)
#   env ONLY="C_mc2 C_mc3" bash slurm/submit_opt_gl_TiNbZrHfTaMoWY.sh   selected tags
#   env DRY=1 bash slurm/submit_opt_gl_TiNbZrHfTaMoWY.sh                print what would be submitted
[ -n "${SLURM_JOB_ID:-}" ] && { echo "this submit script calls sbatch itself: run it on the login node WITHOUT sbatch, e.g.  bash slurm/submit_opt_gl_TiNbZrHfTaMoWY.sh"; exit 2; }
cd "$(dirname "$0")/.." || exit 1
CHAIN=${CHAIN:-3}; ONLY=${ONLY:-}; DRY=${DRY:-0}
DB=databases/sra.TiNbZrHfTaMoWY.kp10.ew0.6.gl.csv
export ELEMENTS=Ti,Nb,Zr,Hf,Ta,Mo,W,Y             # via the environment (--export=ALL), never inside --export=ALL,VAR=a,b
# tag     target    mincomp  beta
RUNS="
C_mc2     Tc_mu0.1  2   2
C_mc3     Tc_mu0.1  3   2
C_mc4     Tc_mu0.1  4   2
C_mc5     Tc_mu0.1  5   2
C_mc6     Tc_mu0.1  6   2
C_mc7     Tc_mu0.1  7   2
C_mc8     Tc_mu0.1  8   2
"
[ -s "${DB}" ] || { echo "missing ${DB} (git pull?)"; exit 2; }
mkdir -p slurm/logs
n=0
while read -r tag target mc beta; do
    [ -z "${tag}" ] && continue
    if [ -n "${ONLY}" ] && [[ " ${ONLY} " != *" ${tag} "* ]]; then continue; fi
    NAME=$(basename "${DB}" .csv).mu0.1.acqbeta${beta}.stonerbeta0.mixdebye.mincomp${mc}.r0.05.max0.7.face0.1
    export NAME INIT_DB=${DB} MINCOMP=${mc} BETA=${beta} TARGET=${target}
    prev=""
    for ((k = 1; k <= CHAIN; k++)); do
        dep=(); [ -n "${prev}" ] && dep=(--dependency=afterany:${prev})
        if [ "${DRY}" = 1 ]; then
            echo "sbatch --job-name=opt_${tag} ${dep[*]} slurm/topola_opt.sbatch   NAME=${NAME} TARGET=${target} MINCOMP=${mc} BETA=${beta} ELEMENTS=${ELEMENTS}"
            prev="<${tag}_${k}>"
        else
            prev=$(sbatch --parsable --export=ALL --job-name="opt_${tag}" "${dep[@]}" slurm/topola_opt.sbatch) || exit 3
            echo "${tag}: job ${prev} (${k}/${CHAIN})  -> opt.ratios.gl/${NAME}"
        fi
    done
    n=$((n + 1))
done <<< "${RUNS}"
echo "${n} runs x ${CHAIN} chained jobs"
