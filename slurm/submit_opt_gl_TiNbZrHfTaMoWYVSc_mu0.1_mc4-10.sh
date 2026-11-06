#!/bin/bash
# GL optimisation runs on set B = Ti,Nb,Zr,Hf,Ta,Mo,W,Y,V,Sc WITHOUT the Tc correction (target Tc_mu0.1, exactly as the
# set-A runs of submit_opt_gl_TiNbZrHfTaMoWYReRu.sh), mincomp 4, 5, 6, 8, 9 and 10 (2026-11-06).
# Completes the uncorrected set-B series: mincomp 2 and 3 come from submit_opt_gl_TiNbZrHfTaMoWYVSc_mc23.sh, mincomp 7
# from _mc7.sh (both finished).  Purpose: set B computed the same way as set A, as a reference for the corr2 runs
# (submit_opt_gl_TiNbZrHfTaMoWYVSc_corr2*.sh), e.g. for a referee question.
# Settings identical to the other set-B runs: init databases/sra.TiNbZrHfTaMoWYVSc.kp10.ew0.6.gl.csv, GL eta, theta_D mix,
# mu* 0.1, Stoner penalty off, max 70 % per element, required components >= 5 %, beta 2, 10 champions per iteration,
# early stop after 15 iterations, seed 100.  Tc_mu0.1 does not depend on the TC_CORR_* constants.
# Output opt.ratios.gl/<init>.mu0.1.acqbeta2.stonerbeta0.mixdebye.mincomp<m>.r0.05.max0.7.face0.1 (same tag as mc 2/3/7),
# logs slurm/logs/opt_<tag>_<jobid>.out.  6 runs x 28 cpus = 168 cpus.  Run on the login node with bash, NOT with sbatch:
#   bash slurm/submit_opt_gl_TiNbZrHfTaMoWYVSc_mu0.1_mc4-10.sh                         all 6 runs, CHAIN=3 (3 x 2 days)
#   env ONLY="B_mc8 B_mc9" bash slurm/submit_opt_gl_TiNbZrHfTaMoWYVSc_mu0.1_mc4-10.sh  selected tags
#   env DRY=1 bash slurm/submit_opt_gl_TiNbZrHfTaMoWYVSc_mu0.1_mc4-10.sh               print what would be submitted
[ -n "${SLURM_JOB_ID:-}" ] && { echo "this submit script calls sbatch itself: run it on the login node WITHOUT sbatch, e.g.  bash slurm/submit_opt_gl_TiNbZrHfTaMoWYVSc_mu0.1_mc4-10.sh"; exit 2; }
cd "$(dirname "$0")/.." || exit 1
CHAIN=${CHAIN:-3}; ONLY=${ONLY:-}; DRY=${DRY:-0}
DB=databases/sra.TiNbZrHfTaMoWYVSc.kp10.ew0.6.gl.csv
export ELEMENTS=Ti,Nb,Zr,Hf,Ta,Mo,W,Y,V,Sc        # via the environment (--export=ALL), never inside --export=ALL,VAR=a,b
# tag     target    mincomp  beta
RUNS="
B_mc4     Tc_mu0.1  4   2
B_mc5     Tc_mu0.1  5   2
B_mc6     Tc_mu0.1  6   2
B_mc8     Tc_mu0.1  8   2
B_mc9     Tc_mu0.1  9   2
B_mc10    Tc_mu0.1  10  2
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
